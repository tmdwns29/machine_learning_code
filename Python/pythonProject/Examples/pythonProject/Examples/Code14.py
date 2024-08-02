import numpy as np
import tensorflow as tf

# mnist 데이터셋에서 훈련데이터와 테스트데이터 분할
(X_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(X_train)
print(y_train)
data_size = X_train.shape[0]
print(data_size) # 60000개

data_size = X_train.shape[0]
batch_size = 12

selected = np.random.choice(data_size, batch_size)
print(selected)
print(X_train[selected])
print('\n\n')
print(y_train[selected])
x_batch = X_train[selected]
y_batch = y_train[selected]