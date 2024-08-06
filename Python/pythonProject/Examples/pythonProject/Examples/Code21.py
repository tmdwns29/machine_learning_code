import FinanceDataReader as fdr
import numpy as np
from matplotlib import pyplot as plt

# 2016-2020
samsung = fdr.DataReader(symbol='005930', start='01/01/2016', end='12/23/2020') # 삼성
print(samsung)
print(samsung.shape)
print(samsung.columns) # pandas dtype (object: 최상위클래스)
print(samsung[['Open']])
open_values = samsung[['Open']]
print(open_values)
print(open_values.shape)

# 2016-최근
samsung = fdr.DataReader(symbol='005930', start='01/01/2016', end=None) # 삼성
print(samsung)
print(samsung.shape)
print(samsung.columns) # pandas dtype (object: 최상위클래스)
print(samsung[['Open']])
open_values = samsung[['Open']]
print(open_values)
print(open_values.shape)
# 데이터가 너무 큼(간격, 값) => 정규화
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)) # 최대값 1, 최소값 0
scaled_open_values = scaler.fit_transform(X=open_values)
print(scaled_open_values)
print(scaled_open_values.shape)

TEST_SIZE = 200
# 훈련 데이터, 테스트 데이터 : 테스트 200
train_data = scaled_open_values[:-TEST_SIZE] # 뒤에서부터 200개
test_data = scaled_open_values[-TEST_SIZE:]
print(train_data)
print(train_data.shape) # (1914)
print(test_data)
print(test_data.shape) # (200)

# RNN - LSTM : 입력데이터 타입이 Tensor 3
def make_feature(open_values, windowing) -> tuple:
    train = list()
    test = list()
    for i in range(len(open_values) - windowing):
        train.append(open_values[i: i+windowing])
        test.append(open_values[i+windowing])
    print(train)
    print(test)
    return np.array(train), np.array(test)

(X_train, y_train) = make_feature(open_values=train_data, windowing=30)
print(f'X_train : {X_train}')
print(f'X_train.shape : {X_train.shape}')
print(f'y_train : {y_train}')
print(f'y_train.shape : {y_train.shape}')

# LSTM으로 구현하기
import keras
model = keras.models.Sequential(name='LSTM_MODEL')
model.add(keras.Input(shape=(X_train.shape[1], 1), name='INPUT'))
# LSTM units=32, cell=32 -> 메모리 개수
model.add(keras.layers.LSTM(units=32, return_sequences=True,
                            activation='tanh', name='LAYER1'))
model.add(keras.layers.LSTM(units=16, return_sequences=False,
                            activation='tanh', name='LAYER2'))
model.add(keras.layers.Dense(units=1, activation='sigmoid', name='OUTPUT'))
model.summary()

# 모델 학습
model.compile(loss='mse', optimizer='adam')
# model.fit(x=X_train, y=y_train, epochs=100, batch_size=16)
# model.save('LSTM_MODEL.keras')

model_2 = keras.models.load_model('LSTM_MODEL.keras')

# 테스트 진행
(X_test, y_test) = make_feature(open_values=test_data, windowing=30)
predictions = model_2.predict(x=X_test)
print(predictions)
print(predictions.shape)

# 그래프로 정답과 예측 확인
plt.figure(figsize=(10,8))
plt.plot(y_test, label='STOCK PRICE', color='blue')
plt.plot(predictions, label='Predictions', color='red')
plt.legend()
plt.show()