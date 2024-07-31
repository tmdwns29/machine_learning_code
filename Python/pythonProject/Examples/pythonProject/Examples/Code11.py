import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

X = [[174],
     [152],
     [138],
     [128],
     [186],]
y = [71, 55, 46, 38, 88] # 행벡터

print(X)
print(y)

X = np.array(X)
y = np.array(y)

lr = LinearRegression()
lr.fit(X=X, y=y)
print(lr.score(X=X, y=y))

y_predict = lr.predict(np.array([[183]]))
print(y_predict) # 67.30998637 키에 따른 예측 몸무게

y_predicts = lr.predict(X)
plt.scatter(X, y_predicts, color='blue', marker='*') # 예측
plt.scatter(X, y, color='red', marker='o') # 정답
plt.show()