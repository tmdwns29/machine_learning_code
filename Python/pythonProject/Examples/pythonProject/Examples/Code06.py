import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# 당뇨병 데이터 가져오기
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes_X)
print(diabetes_X.shape)
print(diabetes_y)
print(diabetes_y.shape)

bmi = diabetes_X[:, np.newaxis, 2] # 축 생성
print(bmi)
print(bmi.shape)

from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(bmi, diabetes_y, test_size=0.2)
print(X_train)
print(X_train.shape)
print(X_test)
print(X_test.shape)