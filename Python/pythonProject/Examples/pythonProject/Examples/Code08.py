import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 닥스훈트 길이, 높이 데이터
dachs_length = [77, 78, 85, 83, 73, 77, 73, 80]
dachs_height = [25, 28, 29, 30, 21, 22, 17, 35]

# 사모예드 길이, 높이 데이터
samo_length = [75, 77, 86, 86, 79, 83, 83, 88]
samo_height = [56, 57, 50, 53, 60, 53, 49, 61]

# 닥스훈트, 사모예드의 산점도
plt.scatter(dachs_length, dachs_height, c='red',
            marker='o', label='Dachshund')
plt.scatter(samo_length, samo_height, c='blue',
            marker='*', label='Samoyed')
plt.xlabel('Length')
plt.ylabel('Height')
plt.title('Dog size')
plt.legend(loc='lower right') # 범례
# plt.show()

# 새로운 개의 데이터
new_dog_length = [79]
new_dog_height = [35]

# 새로운 개가 어디에 근접한지 확인
plt.scatter(new_dog_length, new_dog_height,
            marker='p', c='cyan', label='new dog')
# plt.show()

# 닥스훈트 데이터 | column_stack : 1차원 배열들을 컬럼으로 쌓아 2차원 배열로 만들기
dachs_data = np.column_stack((dachs_length, dachs_height))
print(dachs_data)
print(dachs_data.shape)
dachs_label = np.array([0, 0, 0, 0, 0, 0, 0, 0,])
print(dachs_label)
print(dachs_label.shape)

# 사모예드 데이터
samoyed_data = np.column_stack((samo_length, samo_height))
print(samoyed_data)
print(samoyed_data.shape)
samoyed_data_label = np.ones(len(samoyed_data))
print(samoyed_data_label)
print(samoyed_data_label.shape)

# 새로운 개 데이터 길이 75cm, 높이 35cm
new_dog_data = np.array([[75, 35]])
print(new_dog_data)
print(new_dog_data.shape)

# 닥스, 사모예드 데이터를 행방향으로 결합
dogs_data = np.concatenate((dachs_data, samoyed_data), axis=0)
print(dogs_data)
print(dogs_data.shape)

# 닥스, 사모예드의 예측 데이터들을 결합
dogs_label = np.concatenate((dachs_label, samoyed_data_label))
print(dogs_label)
print(dogs_label.shape)

# 가장 근접한 k개의 데이터들을 바탕으로 예측 성능
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(dogs_data, dogs_label)
print(knn.score(X=dogs_data, y=dogs_label))
y_predicts = knn.predict((dogs_data))
print(dogs_label)

# 새로운 개를 입력으로 넣었을 때의 새로운 예측 데이터
predict_new_dog = knn.predict(X=new_dog_data)
print(predict_new_dog)

# 새로운 개의 예측 분류
classes = {0:'Dachshund', 1:'Samoyed'}
print(f'새로운 강아지는 : {classes[predict_new_dog[0]]}로 예측된다.')

