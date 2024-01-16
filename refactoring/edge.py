import socket as s
import pickle
import keras as k
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

def create_model():
    model = k.Sequential([
    k.layers.Dense(16, activation='relu', input_shape=(1,)),
    k.layers.Dense(16, activation='relu'),
    k.layers.Dense(1)
    ])
    
    return model

class EdgeClient:
    def __init__(self, host='localhost', port=12345):
        self.sock = s.socket(s.AF_INET, s.SOCK_STREAM)
        self.sock.connect((host, port))
        self.model = create_model()
        self.model.compile(optimizer='adam', loss='mse')
        self.losses = []  # 손실값을 저장할 리스트

    def recvall(self, sock):
        data = b''
        while True:
            part = sock.recv(1024)
            data += part
            if len(part) < 1024:
                break
        return data

    def train(self, x_train, y_train, epochs, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs}")
            epoch_loss = 0
            for step, (x_batch, y_batch) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch, training=True)
                    loss = k.losses.MeanSquaredError()(y_batch, y_pred)
                gradients = tape.gradient(loss, self.model.trainable_weights)
                epoch_loss += loss.numpy()  # 에포크별 총 손실값을 기록
                
                # 그래디언트를 서버에 전송합니다.
                data = pickle.dumps(gradients)
                self.sock.send(data)

                # 서버로부터 업데이트된 가중치를 받습니다.
                data = self.recvall(self.sock)
                weights = pickle.loads(data)
                self.model.set_weights(weights)
            
            self.losses.append(epoch_loss / len(dataset))  # 에포크별 평균 손실값을 저장

    def run(self):
        # 서버로부터 x_train의 일부, epochs, batch_size를 받습니다.
        data = self.recvall(self.sock)
        x_train, y_train, epochs, batch_size = pickle.loads(data)

        # 받은 데이터와 설정으로 모델을 학습합니다.
        self.train(x_train, y_train, epochs, batch_size)

        # 학습이 끝난 후 손실 그래프를 그립니다.
        plt.plot(range(1, epochs+1), self.losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

def main():
    ec = EdgeClient()
    ec.run()
    
if __name__ == "__main__":
    main()