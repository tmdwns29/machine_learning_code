import numpy as np
import time
from matplotlib import pyplot as plt

SAMPLE_NUMBER = 10_000 # 수 가독성을 위한 언더바_ 표기 C++:10'000
np.random.seed(int(time.time())) # 랜덤시간을 컴퓨터 시간 값으로 시드를 사용
Xs = np.random.uniform(low=-2.0, high=0.5, size=SAMPLE_NUMBER)
np.random.shuffle(Xs)
print(Xs[:15])
ys = 2 * np.square(Xs) + 3 * Xs + 5

plt.plot(Xs, ys, 'r.')
plt.show()

ys += 0.2 * np.random.randn(SAMPLE_NUMBER)
plt.plot(Xs, ys, 'r.')
plt.show()

import tensorflow as tf

# 비선형 모델 생성
model = tf.keras.Sequential(name='NonLinear_MODEL')

# 입력층 생성
input_layer = tf.keras.Input(shape=(1,))
model.add(input_layer)

# 은닉층 3개 생성
model.add(tf.keras.layers.Dense(units=16, activation='relu', name='LAYER1'))
model.add(tf.keras.layers.Dense(units=8, activation='relu', name='LAYER2'))
model.add(tf.keras.layers.Dense(units=4, activation='relu', name='LAYER3'))

# 출력층 생성
model.add(tf.keras.layers.Dense(units=1, activation='relu', name='OUTPUT'))

# 모델 요약 정보 확인 및 컴파일
model.summary()
model.compile(loss='mse', optimizer='adam')

from sklearn.model_selection import train_test_split

# 훈련 데이터 및 테스트 데이터 분할
(X_train, X_test, y_train, y_test) = train_test_split(Xs, ys, test_size=0.2)

# 훈련 데이터 산점도와 테스트 데이터 산점도
plt.plot(X_train, y_train, 'b.', label='Train')
plt.plot(X_test, y_test, 'r.', label='Test')
plt.legend()
plt.show()

# 모델 학습 500번
model.fit(X_train, y_train, epochs=500)

# 학습된 모델의 예측 값 및 산점도
y_predict = model.predict(X_test)
print(y_predict)
print(y_test)
plt.plot(y_predict, y_test, 'r.')
plt.show()