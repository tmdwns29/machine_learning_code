import numpy as np
from sklearn.linear_model import Perceptron
# 바이어스 지정 안해도 랜덤하게 지정해줌

# 샘플과 레이블이다. (AND연산)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1]) # 활성함수, 계단함수 내장
percetron = Perceptron(tol=0.001, random_state=0)
# tol : 종료조건, 얼마만큼 오차를 허용할 지
percetron.fit(X=X, y=y)
print(percetron.predict(X=X))

# 샘플과 레이블이다. (OR연산)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 1]) # 활성함수, 계단함수 내장
percetron = Perceptron(tol=0.001, random_state=0)
percetron.fit(X=X, y=y)
print(percetron.predict(X=X))

# 샘플과 레이블이다. (XOR연산 : error)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0]) # 활성함수, 계단함수 내장
percetron = Perceptron(tol=0.001, random_state=0)
percetron.fit(X=X, y=y)
print(percetron.predict(X=X))