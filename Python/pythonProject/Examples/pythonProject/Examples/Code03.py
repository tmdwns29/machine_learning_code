import matplotlib.pyplot as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

X = [[0], [1], [2],] # 입력 데이터
y = [3, 3.5, 5.5,] # 출력 : 정답, 레이블, 타겟 | y = wx^T + b -> 직선의 방적식

# 학습을 시킨다. fit()
reg.fit(X=X, y=y) # 모델을 학습시키는 함수
print(reg.score(X=X, y=y)) # 0.8928... -> 학습된 모델의 성능 평가
print(reg.coef_) # [1.25] -> x의 계수 (기울기)
print(reg.intercept_) # 2.750000 (절편)

print(reg.predict(X=[[5]])) # X에 5를 넣었을 때 예상 결과값

plt.scatter(x=X, y=y, color='red') # 빨간 점으로 표현
# plt.show()

y_predict = reg.predict(X=X) # x값에 따른 예측된 y값
plt.plot(X, y_predict, color='blue', linewidth=2)
plt.show()
# y = ax + b -> y = 1.25x + 2.75