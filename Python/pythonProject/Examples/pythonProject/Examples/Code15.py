import tensorflow as tf
import numpy as np

# Sequential모델 : 레이어를 순차적으로 쌓아가는 방식
model = tf.keras.models.Sequential([], name='XOR_MODEL')

# 입력층 생성 -> Input객체
input_layer = tf.keras.layers.Input(shape=(2,), name='INPUT')
model.add(input_layer)

# 은닉층1 추가 -> Dense : 뉴런의 입력과 출력을 연결해주는 역할
layer1 = tf.keras.layers.Dense(units=4, activation='sigmoid', name='LAYER1')
model.add(layer1)

# 은닉층2 추가
layer2 = tf.keras.layers.Dense(units=2, activation='sigmoid', name='LAYER2')
model.add(layer2)

# 출력층 생성
output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', name='OUTPUT')
model.add(output_layer)

model.summary() # 모델을 요약해서 layer마다 shape와 같은 정보 확인
model.compile(optimizer=tf.keras.optimizers.SGD(0.7), loss='mse') # 모델 컴파일

X = tf.constant([[0,0], [0,1], [1,0], [1,1]])
y = tf.constant([0, 1, 1, 0])
model.fit(X, y, epochs=10_000, batch_size=1) # epochs : 학습 횟수
print(np.round(model.predict(X)))