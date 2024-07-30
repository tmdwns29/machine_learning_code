from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

iris = load_iris()
print(iris)
print(type(iris)) # shape안됨 : dictionary타입
print(iris.data) # data의 값 부분
print(type(iris.data)) # data의 값 부분은 넘파이 array타입
print((iris.data).shape)

data = iris.data
target = iris.target
print(target)
print(type(target))
print(target.shape)

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(data, target, train_size=0.8)
print(X_train.shape)
print(X_test.shape)

# 머신러닝 모델 선택하기
regression = LinearRegression() # 생성자 호출
# 훈련 시키기
regression.fit(X_train, y_train)
print(f'Weight : {regression.coef_}') # x의 계수
print(f'Bias : {regression.intercept_}') # y절편
print(regression.score(X_train, y_train)) # 훈련했을 때 맞춘 점수 %수치

# 예측 해보기 (X_test 사용하기)
y_predicts = regression.predict(X=X_test)
# print(y_predicts)
# print(y_predicts.shape)
# print(y_test) # 정답표 출력

import numpy as np
print(np.round(y_predicts).astype(int))
print(y_test)

rounded_y_predicts = np.round(y_predicts).astype(int)
from matplotlib import pyplot as plt
plt.plot(rounded_y_predicts, y_test, 'b.') # 1개 틀림
plt.show()

