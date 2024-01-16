import socket as s
import pickle
import threading
import time
import keras as k
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

#엣지 노드 개수
MAX_EDGE_NODE = 2

# 입력 데이터와 타겟 데이터
x_train = np.arange(-2, 2, 0.01)
y_train = x_train**2

# 데이터의 형태를 [데이터 수, 특성 수]로 변경
x_train = np.reshape(x_train, (-1, 1))
y_train = np.reshape(y_train, (-1, 1))

# 배치 크기와 에폭 수 설정
batch_size = 20
epochs = 100

def create_model():
    model = k.Sequential([
    k.layers.Dense(16, activation='relu', input_shape=(1,)),
    k.layers.Dense(16, activation='relu'),
    k.layers.Dense(1)
    ])
    
    return model

class ParameterServer:
    def __init__(self, host='localhost', port=12345):
        self.sock = s.socket(s.AF_INET, s.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(MAX_EDGE_NODE)
        self.model = create_model()
        self.model.compile(optimizer='adam', loss='mse')

        print('Parameter server listening ...')
        
    def recvall(self, sock):
        data = b''
        while True:
            part = sock.recv(1024)
            data += part
            if len(part) < 1024:
                break
        return data

    def handler(self, clientsock, addr, edge_idx):
        # 첫 연결 시 x_train의 일부, epochs, batch_size 전송
        start = edge_idx * len(x_train) // MAX_EDGE_NODE
        end = (edge_idx+1) * len(x_train) // MAX_EDGE_NODE
        x_train_edge = x_train[start:end]
        y_train_edge = y_train[start:end]
        data = pickle.dumps((x_train_edge, y_train_edge, epochs, batch_size))
        clientsock.send(data)
        
        while True:
            data = self.recvall(clientsock)
            if not data: 
                break
            gradients = pickle.loads(data)
            
            # 받은 그래디언트를 저장합니다.
            self.gradients.append(gradients)
            
            # 모든 클라이언트로부터 그래디언트를 받았다면 평균을 내고 가중치를 업데이트합니다.
            if len(self.gradients) == MAX_EDGE_NODE:
                # 각 레이어에 대한 그래디언트를 따로 평균냅니다.
                avg_gradients = [np.mean([g[i] for g in self.gradients], axis=0) for i in range(len(self.gradients[0]))]
                self.model.optimizer.apply_gradients(zip(avg_gradients, self.model.trainable_weights))
                self.gradients = []

                # 업데이트된 가중치를 모든 클라이언트에게 전송합니다.
                data = pickle.dumps(self.model.get_weights())
                for sock in self.clientsocks:
                    sock.send(data)
        
        clientsock.close()
        # 클라이언트의 학습이 끝났음을 알림
        self.finished_clients += 1

    def run(self):
        self.gradients = []
        self.clientsocks = []
        edge_idx = 0
        self.finished_clients = 0  # 학습이 끝난 클라이언트 수를 추적
        threads = []  # 스레드 목록을 관리
        
        # MAX_EDGE_NODE 수만큼 클라이언트 연결을 받음
        while edge_idx < MAX_EDGE_NODE:
            clientsock, addr = self.sock.accept()
            self.clientsocks.append(clientsock)
            print('Accepted connection from:', addr)
            client_handler = threading.Thread(
                target=self.handler,
                args=(clientsock, addr, edge_idx)
            )
            client_handler.start()
            threads.append(client_handler)  # 스레드를 목록에 추가
            edge_idx += 1
        
        # 모든 클라이언트의 학습이 끝날 때까지 기다림
        while self.finished_clients < MAX_EDGE_NODE:
            time.sleep(1)  # CPU 사용률을 줄이기 위해 간단한 sleep 추가

        # 모든 스레드가 종료될 때까지 기다림
        for thread in threads:
            thread.join()
            
        # 테스트 데이터 생성
        x_test = np.arange(-3, 3, 0.01)
        y_test = x_test**2
        # 모델 예측값 계산
        y_pred = self.model.predict(x_test)
        # 예측값과 실제값을 그래프로 비교
        plt.figure()
        plt.plot(x_test, y_test, label='True')
        plt.plot(x_test, y_pred, label='Predicted')
        plt.legend()
        plt.show()
            
def main():
    ps = ParameterServer()
    ps.run()
    
if __name__ == "__main__":
    main()