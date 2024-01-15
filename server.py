import socket as s
import pickle
import keras as k
from matplotlib import pyplot as plt
import numpy as np

#엣지 노드 개수
MAX_EDGE_NODE = 2

#HOST
HOST = 'localhost'

#PORT
PORT = 12345

#x_train = np.arange(-1,1,0.01)
#x_test = np.arange(-2,2,0.01)
#y_train = x_train**2
#y_test = x_test**2

#global_model.fit(x_train, y_train, epochs=1000, verbose=2, batch_size=20)

#result_y = global_model.predict(x_test)

#plt.plot(x_test,y_test,'b')
#plt.plot(x_test,result_y,'r')
#plt.show()

#print(global_model.get_weights())
def recv_data(sock):
    data = b''
    
    while True:
        part = sock.recv(1024*4)
        data += part
        if len(part) < 1024*4:
            break
    return data

def main():
    #글로벌 모델 정의 (임의로 정함)
    global_model = k.models.Sequential()
    global_model.add(k.layers.Input(1))
    global_model.add(k.layers.Dense(10, activation='tanh'))
    global_model.add(k.layers.Dense(10, activation='tanh'))
    global_model.add(k.layers.Dense(1))
    global_model.compile(optimizer='SGD', loss='mse')
    
    #글로벌 모델 파라미터
    global_weights = global_model.get_weights()
    print(global_weights)
    
    #임시 훈련데이터
    x_train = np.arange(-2,2,0.01)
    y_train = x_train**2
    
    #서버 소켓 설정
    server_socket = s.socket(s.AF_INET, s.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print("[server] Waiting for edge_node connections...")
    #엣지 노드 연결 대기
    edge_nodes = []
    while len(edge_nodes) < MAX_EDGE_NODE:
        node, addr = server_socket.accept()
        edge_nodes.append(node)
        print(f"[server] Accepted connection from: {addr[0]}");

    print("[server] Ready to receive and send data to edge_nodes")
    
    #훈련 데이터 전송 부분
    print("[server] Send training data")
    #엣지 노드의 수 만큼 훈련 데이터셋을 분할 함
    split_num = int(len(x_train) / MAX_EDGE_NODE)
    split_x = [x_train[i * split_num:(i + 1) * split_num] for i in range((len(x_train) + split_num - 1) // split_num)]
    split_y = [y_train[i * split_num:(i + 1) * split_num] for i in range((len(y_train) + split_num - 1) // split_num)]
    i = 0
    #나눠진 훈련 데이터셋을 각각 엣지 노드에게 전송
    for node in edge_nodes:
            res = {"x_train": split_x[i], "y_train": split_y[i]}
            #객체를 직렬화 함
            serialized_res = pickle.dumps(res)
            node.send(serialized_res)
            i += 1
    #잘 전송 되었다면 ack를 받아서 출력
    for node in edge_nodes:
            req = recv_data(node)
            if not req:
                break
            #응답을 역직렬화 함
            data = pickle.loads(req)
            print(f"Received: {data}")
    #글로벌 모델의 파라미터를 전송하여 로컬과 글로벌 모델의 가중치를 동기화 함
    for node in edge_nodes:
            res = {"weights": global_weights}
            #객체를 직렬화 함
            serialized_res = pickle.dumps(res)
            node.send(serialized_res)
    #잘 전송 되었다면 ack를 받아서 출력
    for node in edge_nodes:
            req = recv_data(node)
            if not req:
                break
            #응답을 역직렬화 함
            data = pickle.loads(req)
            print(f"Received: {data}")
    
    #이후 사용자의 명령에 따라서 학습을 진행하고 파라미터를 업데이트 하여 다시 엣지로 전송하는 코드가 들어갈 예정   
    while True:
        for node in edge_nodes:
            res = {"message": "data"}
            serialized_res = pickle.dumps(res)
            node.send(serialized_res)
        
        for node in edge_nodes:
            req = recv_data(node)
            
            if not req:
                break
            
            data = pickle.loads(req)
            print(f"Received: {data}")
        
        user_input = input()
        if user_input.lower() == 'q':
            print("[server] Closing server...")
            server_socket.close()    
            break
    #모든 학습이 끝난 후 러닝 커브와 얼마나 훈련이 됐는지 그래프를 보여줄 코드가 작성될 부분
        
if __name__ == "__main__":
    main()
