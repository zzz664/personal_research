import socket as s
import pickle
import keras as k
import tensorflow as tf
from matplotlib import pyplot as plt
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

def create_model():
    model = k.models.Sequential()
    model.add(k.layers.Input(1))
    model.add(k.layers.Dense(32, activation='tanh'))
    model.add(k.layers.Dense(32, activation='tanh'))
    model.add(k.layers.Dense(1))
    return model

def compute_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = k.losses.MeanSquaredError()(y, y_pred)
    return tape.gradient(loss, model.trainable_variables)

def apply_gradients(model, gradients):
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
def main():
    #로컬 모델 정의
    local_model = create_model()
    local_model.compile(optimizer='SGD', loss='mse')
    
    local_epochs = 0
    local_batch_size = 0

    #클라이언트 소켓 설정
    client = s.socket(s.AF_INET, s.SOCK_STREAM)
    client.connect((HOST, PORT))
    
    #서버로부터 훈련 데이터셋을 받음
    res = recv_data(client)
    #데이터 역직렬화
    data = pickle.loads(res)
    print("[client] Received: training data")
    local_x_train = data["x_train"]
    local_y_train = local_x_train**2
    local_epochs = data["epochs"]
    local_batch_size = data["batch_size"]
    print(f"[client] Epochs : {local_epochs} / Batch size : {local_batch_size}")
    
    #잘 받았음을 알리는 ack를 전송
    req = {"client_msg":"ack(success recv training data)"}
    #응답 직렬화
    serialized_req = pickle.dumps(req)
    client.send(serialized_req)
    
    #서버로부터 글로벌 모델의 파라미터를 받음
    res = recv_data(client)
    #데이터 역직렬화
    data = pickle.loads(res)
    print("[client] Received: global weights")
    
    #받아온 글로벌 모델의 파라미터로 로컬 모델을 동기화 함
    print("[client] Synchronize with global model weights")
    local_model.set_weights(data["weights"])
    local_model_weights = None
    
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
        
        if data["msg_type"] == "start":
            print("[client] Received start learning message")
            #local_model.fit(local_x_train, local_y_train, epochs=local_epochs, verbose=1, batch_size=local_batch_size)
            for epoch in range(0, local_epochs):
                print("에폭:"+str(epoch))
                batch_count = len(local_x_train) // local_batch_size
                for i in range(0, batch_count):
                    #print("배치:"+str(i)+" "+str(batch_count))
                    x_batch = local_x_train[i * local_batch_size:(i + 1) * local_batch_size]
                    y_batch = local_y_train[i * local_batch_size:(i + 1) * local_batch_size]
                    
                    grads = compute_gradients(local_model, x_batch, y_batch)
                    
                    req = {"client_msg":"ack(share gradients)", "grads": grads}
                    serialized_req = pickle.dumps(req)
                    client.send(serialized_req)
                    
                    res = recv_data(client)
                    if not res:
                        break
                    data = pickle.loads(res)
                    
                    apply_gradients(local_model, data["grads"])
                
                #local_model_weights = local_model.get_weights()
                #
                #req = {"msg_type":"update", "client_msg":"ack(send local weights)"
                #   , "weights": local_model_weights}
                #serialized_req = pickle.dumps(req)
                #client.send(serialized_req)
                #
                #res = recv_data(client)
                #if not res:
                #    break
                #data = pickle.loads(res)
                #if data["msg_type"] == "update":
                #    print("[client] Received: global weights")
                #    print("[client] Synchronize with global model weights")
                #    local_model.set_weights(data["weights"])
    #
                #    #잘 받았음을 알리는 ack를 전송
                #    req = {"client_msg":"ack(success recv global weights)"}
                #    serialized_req = pickle.dumps(req)
                #    client.send(serialized_req)
                    
            req = {"client_msg":"ack(success learning)"}
            serialized_req = pickle.dumps(req)
            client.send(serialized_req)
            
            print("[client] Success learning")
            
            
            x_test = np.arange(-2,2,0.01)
            y_test = x_test**2
            result_y = local_model.predict(x_test)
            plt.plot(x_test,y_test,'b')
            plt.plot(x_test,result_y,'r')
            plt.show()
        
        elif data["msg_type"] == "update":
            print("[client] Received: global weights")
            print("[client] Synchronize with global model weights")
            local_model.set_weights(data["weights"])
    
            #잘 받았음을 알리는 ack를 전송
            req = {"client_msg":"ack(success recv global weights)"}
            serialized_req = pickle.dumps(req)
            client.send(serialized_req)
            
        elif data["msg_type"] == "setting":
            print("[client] Received setting data")
            local_epochs = data["epochs"]
            local_batch_size = data["batch_size"]
        
            req = {"client_msg":"ack(" + data["msg_type"] + ")"}
            serialized_req = pickle.dumps(req)
            client.send(serialized_req)
        
        else:
            print("[client] Received msg")
            req = {"client_msg":"ack(" + data["msg_type"] + ")"}
            serialized_req = pickle.dumps(req)
            client.send(serialized_req)
    
    client.close()
    
if __name__ == "__main__":
    main()