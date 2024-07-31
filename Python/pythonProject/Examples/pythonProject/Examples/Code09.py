import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 닥스훈트 길이, 높이 데이터
dachs_length = [77, 78, 85, 83, 73, 77, 73, 80]
dachs_height = [25, 28, 29, 30, 21, 22, 17, 35]

# 사모예드 길이, 높이 데이터
samo_length = [75, 77, 86, 86, 79, 83, 83, 88]
samo_height = [56, 57, 50, 53, 60, 53, 49, 61]

# 새로운 개의 데이터
new_dog_length = [79]
new_dog_height = [35]

dachs_data = np.column_stack((dachs_length, dachs_height))
print(dachs_data)
print(dachs_data.shape)
dachs_label = np.array([0, 0, 0, 0, 0, 0, 0, 0,])
print(dachs_label)
print(dachs_label.shape)

samoyed_data = np.column_stack((samo_length, samo_height))
print(samoyed_data)
print(samoyed_data.shape)

samoyed_data_label = np.ones(len(samoyed_data))
print(samoyed_data_label)
print(samoyed_data_label.shape)

new_dog_data = np.array([[75, 35]])
print(new_dog_data)
print(new_dog_data.shape)

dogs_data = np.concatenate((dachs_data, samoyed_data), axis=0)
print(dogs_data)
print(dogs_data.shape)
dogs_label = np.concatenate((dachs_label, samoyed_data_label))
print(dogs_label)
print(dogs_label.shape)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(dogs_data, dogs_label)
print(knn.score(X=dogs_data, y=dogs_label))
y_predicts = knn.predict((dogs_data))
print(dogs_label)

predict_new_dog = knn.predict(X=new_dog_data)
print(predict_new_dog)

classes = {0:'Dachshund', 1:'Samoyed'}
print(f'새로운 강아지는 : {classes[predict_new_dog[0]]}로 예측된다.')

# 통계적 기반으로 데이터 증강해보기

# 닥스훈트
dachs_length_mean = np.mean(dachs_length)
dachs_height_mean = np.mean(dachs_height)
print(f'닥스훈트 길이의 평균 : {dachs_length_mean}\tcm')
print(f'닥스훈트 높이의 평균 : {dachs_height_mean}\tcm')
new_normal_dachs_length = np.random.normal(dachs_length_mean, 8.0, 200)
new_normal_dachs_height = np.random.normal(dachs_height_mean, 8.0, 200)
print(new_normal_dachs_length)
print(new_normal_dachs_length.shape)
print(new_normal_dachs_height)
print(new_normal_dachs_height.shape)

# 사모예드
samo_length_mean = np.mean(samo_length)
samo_height_mean = np.mean(samo_height)
print(f'사모예드 길이의 평균 : {samo_length_mean}\tcm')
print(f'사모예드 높이의 평균 : {samo_height_mean}\tcm')
new_normal_samo_length = np.random.normal(samo_length_mean, 8.0, 200)
new_normal_samo_height = np.random.normal(samo_height_mean, 8.0, 200)
print(new_normal_samo_length)
print(new_normal_samo_length.shape)
print(new_normal_samo_height)
print(new_normal_samo_height.shape)

plt.scatter(new_normal_dachs_length, new_normal_dachs_height, c='b', marker='+')
plt.scatter(new_normal_samo_length, new_normal_samo_height, c='r', marker='*')
plt.show()

# 새로운 데이터 합성하기
new_dachs_data = np.column_stack((new_normal_dachs_length, new_normal_dachs_height))
new_samo_data = np.column_stack((new_normal_samo_length, new_normal_samo_height))

# 새로운 레이블 합성하기
new_dachs_label = np.zeros(len(new_dachs_data)) # 200
new_samo_label = np.ones(len(new_samo_data)) # 200
print(new_dachs_data)
print(new_dachs_data.shape)
print(new_samo_data)
print(new_samo_data.shape)
print(new_dachs_label)
print(new_samo_label)

# 총 400개의 개 데이터 만들기
new_dogs = np.concatenate((new_dachs_data, new_samo_data))
new_labels = np.concatenate((new_dachs_label, new_samo_label))
print(new_dogs.shape)
print(new_labels.shape)

# 훈련데이터 + 테스트데이터 분리하기
(X_train, X_test, y_train, y_test) = train_test_split(new_dogs, new_labels, test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

k = 7
knn = KNeighborsClassifier(k)
knn.fit(X=X_train, y=y_train)
print(f'훈련 정확도 : {knn.score(X_train, y_train)}')

# 예측하기
y_predicts2 = knn.predict(X_test)
print(y_predicts2)
print(y_test)

print(f'테스트 정확도 : {accuracy_score(y_test, y_predicts2)}')
plt.plot(y_predicts2, y_test, 'b')
plt.show()