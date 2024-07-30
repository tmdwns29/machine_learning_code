import pandas as pd

wine = pd.read_csv(filepath_or_buffer="wine_csv_data.csv")
print(wine)
print(wine.head())
print(wine.shape)
print(wine.info())

data = wine[['alcohol', 'sugar', 'pH']].to_numpy() # 데이터
print(data)
print(data.shape)
target = wine['class'].to_numpy() # 정답
print(target)
print(target.shape)

from sklearn.model_selection import train_test_split # 패키지

(X_train, X_test, y_train, y_test) = train_test_split(data, target, test_size=0.2)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X=X_train, y=y_train)
print(lr.score(X=X_train, y=y_train))
print(lr.score(X=X_test, y=y_test))
print(lr.coef_)
print(lr.intercept_)
# weight 값 찾는 것이 중요
# 1. 다량의 데이터의 개수
# 2. 정규화
# 3. 규제화 norm
# 4. 데이터 증감
# 훈련데이터 : 테스트데이터 = 8:2 or 7:3

# 95p : 선형회귀 예제

# cost function : 비용함수-뭔가를 하기위한 손해(미분)
# 학습 데이터는 반드시 2차원 배열이어야 함