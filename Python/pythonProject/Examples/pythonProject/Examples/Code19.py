import tensorflow as tf
import numpy as np
import keras

fasion_mnist = keras.datasets.fashion_mnist
print(fasion_mnist) # keras 데이터타입
(X_train, y_train), (X_test, y_test) = fasion_mnist.load_data()
print(X_train)
print(X_train.shape)
print(X_test.shape)
X_train_CNN = X_train.reshape((-2, 28, 28, 1)) / 255.0 # 0.0 ~ 1.0 (60,000)
X_test_CNN = X_test.reshape((-1, 28, 28, 1)) / 255.0 # (10,000)

# CNN 모델 생성
model_CNN = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3),
                        padding='same', activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=2),

    keras.layers.Conv2D(filters=64, kernel_size=(3,3),
                        padding='same', activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=2),

    keras.layers.Conv2D(filters=32, kernel_size=(3,3),
                        padding='same', activation='relu',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
    # Convolution
    # DNN => Flatten()
    keras.layers.Flatten(), # Fully connected
    keras.layers.Dense(units=64, activation='relu', name='LAYER1'),
    keras.layers.Dense(units=32, activation='relu', name='LAYER2'),
    keras.layers.Dense(units=16, activation='relu', name='LAYER3'),
    keras.layers.Dense(units=10, activation='softmax', name='OUTPUT'),
], name='FASION_CNN')

# model_CNN.summary()
# model_CNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
# model_CNN.fit(x=X_train_CNN, y=y_train, epochs=200)
# model_CNN.save('2024-08-05_CNN.keras')
model2 = keras.models.load_model('2024-08-05_CNN.keras')
y_predicts = model2.predict(X_test_CNN)
print(f'y_predicts(예측) : {y_predicts[0]}')
print(f'y_test(정답)     : {y_test[0]}')

print(f'y_predicts(예측) : {np.round(y_predicts[0])}')
print(f'y_test(정답)     : {y_test[0]}')

import time
for i in range(30):
    print(f'{i + 1}번째 예측 : {np.round(y_predicts[i])}')
    print(f'{i + 1}정답     : {y_test[i]}')
    time.sleep(3.0)