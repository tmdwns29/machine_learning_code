import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(X_train)
print(X_train.shape)
print(X_test)
print(X_test.shape)

X_train = np.divide(X_train, 255.0)
X_test = np.divide(X_test, 255.0)

model = keras.models.Sequential(layers=None, name='DIGITS')
# input = keras.Input(shape=(28*28,))
# model.add(input)

# 입력층
input = keras.layers.Flatten(input_shape=(28, 28))
model.add(input)

# 은닉층
layer1 = keras.layers.Dense(units=64, activation='relu', name='LAYER1')
model.add(layer1)
layer2 = keras.layers.Dense(units=32, activation='relu', name='LAYER2')
model.add(layer2)
layer3 = keras.layers.Dense(units=16, activation='relu', name='LAYER3')
model.add(layer3)

# 출력층
layer4 = keras.layers.Dense(units=10, activation='softmax', name='OUTPUT') # softmax
model.add(layer4)

model.add(keras.layers.Dropout(rate=0.2))

model.summary()
print(y_test)
print(y_test.shape)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.8), # 'Adam'
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)