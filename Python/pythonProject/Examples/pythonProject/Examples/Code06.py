import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 당뇨병 데이터 가져오기
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes_X)
print(diabetes_X.shape)
print(diabetes_y)
print(diabetes_y.shape)

# bmi만 추려내서 2차원 배열로 만들기. bmi 특징의 인덱스가 2이다.
bmi = diabetes_X[:, np.newaxis, 2] # 축 생성
print(bmi)
print(bmi.shape)

# 학습용 데이터와 테스트용 데이터 분할
(X_train, X_test, y_train, y_test) = train_test_split(bmi, diabetes_y, test_size=0.1,
                                                      random_state=0)
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
print(regression.score(X_train, y_train))

# 그래프 그리기
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_predicts, color='blue', linewidth=3)
plt.show()
