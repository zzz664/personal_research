import socket as s
import pickle
import threading
import time
import keras as k
from matplotlib import pyplot as plt
import numpy as np

#엣지 노드 개수
MAX_EDGE_NODE = 3

# 입력 데이터와 타겟 데이터
x_train = np.arange(-2, 2, 0.01)
y_train = x_train**2

# 데이터의 형태를 [데이터 수, 특성 수]로 변경
x_train = np.reshape(x_train, (-1, 1))
y_train = np.reshape(y_train, (-1, 1))

# 배치 크기와 에폭 수 설정
batch_size = 20
epochs = 100

# 입력값의 제곱의 결과를 예측하는 간단한 DNN 모델 생성 함수
# 1개의 입력뉴런이 있는 입력층과 각각 16개의 뉴런과 relu 활성화 함수로 구성된 은닉층 2개 1개의 출력뉴런으로 구성
def create_model():
    model = k.Sequential([
    k.layers.Dense(16, activation='relu', input_shape=(1,)),
    k.layers.Dense(16, activation='relu'),
    k.layers.Dense(1)
    ])
    
    return model

#파라미터 서버의 클래스 선언
class ParameterServer:
    #소켓과 각종 변수 및 모델을 초기화
    def __init__(self, host='localhost', port=12345):
        self.gradients = []         #노드들로부터 받는 Gradients저장 리스트
        self.losses = []            #노드들로부터 받는 훈련손실데이터 저장 리스트
        self.mse_values = []        #노드들로부터 받는 검증손실데이터 저장 리스트
        self.clientsocks = []       #클라이언트 소켓 저장 리스트
        self.node_addrs = []        #노드 주소,포트 정보 저장 리스트
        self.finished_clients = 0   #학습이 끝난 클라이언트 수
        
        self.sock = s.socket(s.AF_INET, s.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(MAX_EDGE_NODE)
        self.model = create_model()
        #연속형 변수를 입력받는 회귀 모델이므로 손실함수는 mse로 채택, SGD를 개선한 adam 옵티마이저 채택
        self.model.compile(optimizer='adam', loss='mse')

        print('Parameter server listening ...')
    
    #1024바이트가 넘는 데이터를 받는 경우를 상정하여 1024바이트보다 큰 데이터도 받을 수 있도록 recv함수를 개선
    def recvall(self, sock):
        data = b''
        while True:
            part = sock.recv(1024)
            data += part
            if len(part) < 1024:
                break
        return data

    #각 클라이언트가 연결될 때 생성되는 스레드의 메인 함수
    def handler(self, clientsock, edge_idx):
        #각 노드에게 다른 데이터셋을 보내기 위해 인덱스를 나누는 작업을 진행
        start = edge_idx * len(x_train) // MAX_EDGE_NODE
        end = (edge_idx+1) * len(x_train) // MAX_EDGE_NODE
        #각 노드에게 보낼 데이터셋을 분할
        x_train_edge = x_train[start:end]
        y_train_edge = y_train[start:end]
        #분할한 데이터를 직렬화해서 클라이언트에게 보낸다
        data = pickle.dumps((x_train_edge, y_train_edge, epochs, batch_size))
        clientsock.send(data)
        
        #학습하는 동안 노드간 Gradient 공유를 위해서 무한 루프 설정
        while True:
            #클라이언트로부터 Gradient를 받거나 학습이 끝난 후 검증손실과 트레이닝손실 데이터를 받는 분기처리
            data = self.recvall(clientsock)
            #데이터가 더 이상 오지 않는다면 학습이 끝난걸로 간주하여 무한 루프를 종료
            if not data: 
                break
            #데이터를 역직렬화한다
            unpack = pickle.loads(data)
            #여기서는 분기처리를 위해 데이터 맨 앞에 type을 같이 보내서 서버에서 확인하고 구별할 수 있게함
            if unpack["type"] == "loss_mse" :
                self.losses.append(unpack["loss"])
                self.mse_values.append(unpack["mse"])
                #손실데이터를 보내왔다면 마찬가지로 학습이 종료된 것이나 마찬가지이므로 무한 루프를 종료함
                break
            elif unpack["type"] == "gradients":
                gradients = unpack["gradients"]
            
            # 받은 그래디언트를 저장합니다.
            self.gradients.append(gradients)
            
            # 모든 클라이언트로부터 그래디언트를 받았다면 평균을 내고 가중치를 업데이트
            # 서버에 저장된 노드들의 Gradients 데이터가 최대 노드 연결 수와 같을 때 가중치를 전송함으로서
            # 모든 노드가 동시에 처리할 수 있도록 동기화를 했음
            if len(self.gradients) == MAX_EDGE_NODE:
                # 각 레이어에 대한 Gradient를 따로 평균을 내는 작업
                avg_gradients = [np.mean([g[i] for g in self.gradients], axis=0) for i in range(len(self.gradients[0]))]
                # 계산된 평균 Gradients로 글로벌 모델의 가중치를 업데이트
                self.model.optimizer.apply_gradients(zip(avg_gradients, self.model.trainable_weights))
                self.gradients = []

                # 글로벌 모델의 가중치를 받아서 직렬화 함
                data = pickle.dumps(self.model.get_weights())
                # 업데이트된 가중치를 모든 클라이언트에게 전송
                for sock in self.clientsocks:
                    sock.send(data)
        
        # 무한 루프가 종료되면 학습이 끝났으므로 클라이언트와의 연결을 해제함
        clientsock.close()
        
        # 학습이 끝난 노드의 개수를 하나 증가시킨다
        self.finished_clients += 1
        
    def run(self):
        edge_idx = 0    # 노드들의 구분을 위한 id값
        threads = []    # 스레드 목록을 관리

        # MAX_EDGE_NODE 수만큼 클라이언트 연결을 받음
        while edge_idx < MAX_EDGE_NODE:
            clientsock, addr = self.sock.accept()
            self.clientsocks.append(clientsock)
            print('Accepted connection from:', addr)
            self.node_addrs.append(addr)
            client_handler = threading.Thread(
                target=self.handler,
                args=(clientsock, edge_idx)
            )
            client_handler.start()
            threads.append(client_handler)  # 스레드를 목록에 추가
            edge_idx += 1
        
        # 모든 클라이언트의 학습이 끝날 때까지 기다림
        # CPU 사용률을 줄이기 위해 간단한 sleep 추가
        while self.finished_clients < MAX_EDGE_NODE:
            time.sleep(1)  

        # 모든 스레드가 종료될 때까지 기다림
        for thread in threads:
            thread.join()
            
        # 테스트 데이터 생성
        x_test = np.arange(-2, 2, 0.01)
        y_test = x_test**2
        # 모델 예측값 계산
        y_pred = self.model.predict(x_test)
        # 예측값과 실제값을 그래프로 비교
        plt.figure("[server] predict test")
        plt.plot(x_test, y_test, label='True')
        plt.plot(x_test, y_pred, label='Predicted')
        plt.legend()

        #각 노드들의 검증손실 및 훈련손실값을 그래프로 그림
        for i in range(MAX_EDGE_NODE):
            plt.figure(f"Edge Client host:{self.node_addrs[i][0]} port:{self.node_addrs[i][1]}")
            plt.plot(range(1, epochs+1), self.losses[i], label='Training Loss')
            plt.plot(self.mse_values[i], label='Validation Loss')
            plt.title(f'port {self.node_addrs[i][1]} Training Loss & Validation Loss Graph')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        #모든 그래프 출력
        plt.show()
            
def main():
    ps = ParameterServer()
    ps.run()
    
if __name__ == "__main__":
    main()