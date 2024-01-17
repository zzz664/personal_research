import os
import keras as k
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

# 입력 데이터와 타겟 데이터
x_train = np.arange(-2, 2, 0.01)
y_train = x_train**2

# 데이터의 형태를 [데이터 수, 특성 수]로 변경
x_train = np.reshape(x_train, (-1, 1))
y_train = np.reshape(y_train, (-1, 1))

# 배치 크기와 에폭 수 설정
batch_size = 200
epochs = 1000

# 배치 생성
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

model = k.Sequential([
    k.layers.Dense(16, activation='relu', input_shape=(1,)),
    k.layers.Dense(16, activation='relu'),
    k.layers.Dense(16, activation='relu'),
    k.layers.Dense(16, activation='relu'),
    k.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 검증 데이터
x_val = np.arange(-5, 5, 0.01)
y_val = x_val**2
x_val = np.reshape(x_val, (-1, 1))
y_val = np.reshape(y_val, (-1, 1))

# 학습률 스케줄링
lr_schedule = k.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = k.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = k.losses.MeanSquaredError()

# 학습 루프
for epoch in range(epochs):
    print("\nEpoch {}/{}".format(epoch+1, epochs))
    for step, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # 검증 손실 계산
    val_pred = model(x_val)
    val_loss = loss_fn(y_val, val_pred)
    print("Validation loss: ", val_loss.numpy())
    
x_test = np.arange(-3,3,0.01)
y_test = x_test**2
result_y = model.predict(x_test)
plt.plot(x_test,y_test,'b')
plt.plot(x_test,result_y,'r')
plt.show()