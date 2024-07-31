import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 당뇨병 데이터 가져오기
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes_X)
print(diabetes_X.shape)
print(diabetes_y)
print(diabetes_y.shape)

# bmi 열만 가져오기 (인덱스 2인 값)
bmi = diabetes_X[:, np.newaxis, 2] # 축 생성
print(bmi)
print(bmi.shape)

# 학습용 데이터, 테스트용 데이터 분할
(X_train, X_test, y_train, y_test) = train_test_split(bmi, diabetes_y, test_size=0.2)
print(X_train)
print(X_train.shape)
print(X_test)
print(X_test.shape)

regression = LinearRegression() # 클래스 : 생성자 호출
regression.fit(X_train, y_train)
# 테스트
y_predicts = regression.predict(X_test)
print(y_predicts)
print(y_test)

# 그래프 그리기
plt.plot(y_test, y_predicts, 'b.')
plt.show()
