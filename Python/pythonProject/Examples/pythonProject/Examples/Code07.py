from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics

# 붓꽃 데이터셋 가져오기
iris = load_iris()
print(iris)
print(iris.data)
print(iris.target)

# 붓꽃 데이터 셋의 특징이름을 열로 지정하여 데이터프레임으로 변환
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# target열을 새로 만들고 붓꽃의 target시리즈를 추가
iris_df['target'] = pd.Series(iris.target)
print(iris_df)
print(iris_df.head())
print(iris_df['target'].value_counts())

# 학습데이터, 훈련데이터 분할
(X_train, X_test, y_train, y_test) = train_test_split(iris.data, iris.target, test_size=0.2)
print(X_train)
print(X_train.shape)


# k값이 작을수록 학습에 최적화가 잘됨 k=3일 때 96%(과대적합)
# 3보다는 5,7이 나음 / k값은 홀수가 좋음 / 5가 좋다
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_predicts = knn.predict(X_test)
print(knn.score(X_train, y_train))
print(y_predicts)
print(y_test)
print()

score = metrics.accuracy_score(y_test, y_predicts)
print(score)

classes = {0:'SETOSA', 1:'VERSICOLOR', 2:'VIRGINICA'}
found_new_iris = np.array([[4.0, 2.0, 1.3, 0.4]])
index = knn.predict(found_new_iris)
print(knn.predict(found_new_iris))
print(classes[index[0]])

