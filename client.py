import socket as s
import pickle
import keras as k
import numpy as np

HOST = 'localhost'
PORT = 12345

def recv_data(sock):
    data = b''
    
    while True:
        part = sock.recv(1024*4)
        data += part
        if len(part) < 1024*4:
            break
    return data

def main():
    #로컬 모델 정의
    local_model = k.models.Sequential()
    local_model.add(k.layers.Input(1))
    local_model.add(k.layers.Dense(10, activation='tanh'))
    local_model.add(k.layers.Dense(10, activation='tanh'))
    local_model.add(k.layers.Dense(1))
    local_model.compile(optimizer='SGD', loss='mse')

    #클라이언트 소켓 설정
    client = s.socket(s.AF_INET, s.SOCK_STREAM)
    client.connect((HOST, PORT))
    
    #서버로부터 훈련 데이터셋을 받음
    res = recv_data(client)
    #데이터 역직렬화
    training_data = pickle.loads(res)
    print("[client]Received: training data")
    
    #잘 받았음을 알리는 ack를 전송
    req = {"client_msg":"ack(success recv training data)"}
    #응답 직렬화
    serialized_req = pickle.dumps(req)
    client.send(serialized_req)
    
    #서버로부터 글로벌 모델의 파라미터를 받음
    res = recv_data(client)
    #데이터 역직렬화
    data = pickle.loads(res)
    print("[client]Received: global weights")
    
    #받아온 글로벌 모델의 파라미터로 로컬 모델을 동기화 함
    print("[client]Synchronize with global model weights")
    local_model.set_weights(data["weights"])
    
    #잘 받았음을 알리는 ack를 전송
    req = {"client_msg":"ack(success recv global weights)"}
    serialized_req = pickle.dumps(req)
    client.send(serialized_req)
    
    #이후 학습을 진행하고 서버와 통신하여 파라미터를 업데이트 하는 과정의 코드가 들어갈 예정
    while True:
        res = recv_data(client)
        
        if not res:
            break
        
        data = pickle.loads(res)
        print(f"[client]Received: {data}")
        
        req = {"client_msg":"ack"}
        serialized_req = pickle.dumps(req)
        client.send(serialized_req)
    
    client.close()
    
if __name__ == "__main__":
    main()