import socket as s
import pickle
import keras as k
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np

# 입력값의 제곱의 결과를 예측하는 간단한 DNN 모델 생성 함수
# 1개의 입력뉴런이 있는 입력층과 각각 16개의 뉴런과 relu 활성화 함수로 구성된 은닉층 2개 1개의 출력뉴런으로 구성
# 서버와 동일한 모델
def create_model():
    model = k.Sequential([
    k.layers.Dense(16, activation='relu', input_shape=(1,)),
    k.layers.Dense(16, activation='relu'),
    k.layers.Dense(1)
    ])
    
    return model

#엣지 클라이언트의 클래스 선언
class EdgeClient:
    #소켓과 각종 변수 및 모델을 초기화
    def __init__(self, host='localhost', port=12345):
        self.losses = []        # 손실값을 저장할 리스트
        self.mse_values = []    # Epoch당 MSE를 저장할 리스트
        
        self.sock = s.socket(s.AF_INET, s.SOCK_STREAM)
        self.sock.connect((host, port))
        self.model = create_model()
        #연속형 변수를 입력받는 회귀 모델이므로 손실함수는 mse로 채택, SGD를 개선한 adam 옵티마이저 채택
        self.model.compile(optimizer='adam', loss='mse')
        
    #1024바이트가 넘는 데이터를 받는 경우를 상정하여 1024바이트보다 큰 데이터도 받을 수 있도록 recv함수를 개선
    def recvall(self, sock):
        data = b''
        while True:
            part = sock.recv(1024)
            data += part
            if len(part) < 1024:
                break
        return data

    #Gradient 공유를 위해 Keras에서 제공하는 fit 함수를 사용하는 대신
    #직접 fit 함수의 기능을 구현하여 서버와 통신하여 가중치를 업데이트 할 수 있도록 코드를 작성했다
    def train(self, x_train, y_train, epochs, batch_size):
        #데이터셋을 batch size씩 분할
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

        #설정된 epoch 수 만큼 반복 진행
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0  #Epoch당 훈련손실값을 계산하기위한 변수
            y_preds = []    #Epoch당 검증손실값을 계산하기위한 리스트
            #배치의 개수만큼 반복 진행
            for step, (x_batch, y_batch) in enumerate(dataset):
                #tensorflow에서 제공하는 GradientTape을 통해 모델의 Gradients값을 계산하는 과정
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch, training=True)                 # 예측 값 저장
                    y_preds.extend(y_pred.numpy())                              # 예측 값을 리스트에 추가
                    loss = k.losses.MeanSquaredError()(y_batch, y_pred)         # 예측 값과 실제 값으로 MSE함수를 이용해 손실 계산
                gradients = tape.gradient(loss, self.model.trainable_weights)   #계산된 손실과 가중치로 Gradients를 계산
                epoch_loss += loss.numpy()  # Epoch당 총 손실값을 기록
                
                #Gradients 직렬화
                data = pickle.dumps({"type":"gradients", "gradients":gradients})
                #Gradients를 서버에 전송
                self.sock.send(data)

                #서버로부터 업데이트된 가중치를 받는다.
                data = self.recvall(self.sock)
                #가중치 역직렬화
                weights = pickle.loads(data)
                #로컬 모델의 가중치를 업데이트
                self.model.set_weights(weights)
            
            #Epoch의 검증손실 값 계산
            mse = mean_squared_error(y_train, np.array(y_preds))    # MSE 계산
            self.mse_values.append(mse)                             # MSE 값을 리스트에 추가
            self.losses.append(epoch_loss / len(dataset))           # Epoch당 훈련손실값을 저장

    def run(self):
        #서버로부터 훈련데이터, epochs, batch_size를 받음
        data = self.recvall(self.sock)
        #받은 데이터를 역직렬화
        x_train, y_train, epochs, batch_size = pickle.loads(data)

        #받은 데이터와 설정으로 모델을 학습함
        self.train(x_train, y_train, epochs, batch_size)

        #학습이 끝난 후 데이터 타입을 포함하여 손실 데이터를 직렬화하여 전송
        data = pickle.dumps({"type":"loss_mse", "loss":self.losses, "mse":self.mse_values})
        self.sock.send(data)

def main():
    ec = EdgeClient()
    ec.run()
    
if __name__ == "__main__":
    main()