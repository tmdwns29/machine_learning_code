import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
titanic = sns.load_dataset(name='titanic')
print(titanic)
print(titanic.shape)

titanic.to_csv(path_or_buf='titanic.csv', index=False)
print(titanic.isnull().sum()) # 각 변수마다 NaN의 개수

print(titanic['age']) # 채우기 전
titanic['age'] = titanic['age'].fillna(value=titanic['age'].median())
print(titanic['age']) # 채운 후

print(titanic['embarked'].value_counts())
titanic['embarked'] = titanic['embarked'].fillna(value='S')
print(titanic['embarked'])

print(titanic['embark_town'].value_counts())
titanic['embark_town'] = titanic['embark_town'].fillna(value='Southampton')
print(f"titanic['embark_town'] :\r\n{titanic['embark_town']}")

print(titanic['deck'].value_counts())
titanic['deck'] = titanic['deck'].fillna(value='C') # 앞에서 나온 최빈값으로 치환

print(titanic['deck']) # 비어있는 값 교체 전
titanic['deck'] = titanic['deck'].fillna(value='C')
print(titanic['deck'])

# missing value 다 치환되었는지 확인하는 코드
print(titanic.isnull().sum())

print(titanic.info())

# 타이타닉의 생존자 정보
print(titanic['survived'].value_counts())
# 0 죽음, 1 생존

(f, ax) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
titanic['survived'][titanic['sex'] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
titanic['survived'][titanic['sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)

ax[0].set_title('Survived(Male)')
ax[1].set_title('Survived(FeMale)')
plt.show()

sns.countplot(x='pclass', hue='survived', data=titanic)
plt.title(label='Pclass vs Survived')
plt.show()

# 오류
# titanic_correlation_analysis = titanic.corr(method='pearson')
# print(f'Pearson correlation : \r\n{titanic_correlation_analysis}')

print(titanic['survived'].corr(other=titanic['adult_male'], method='pearson'))
print(titanic['survived'].corr(other=titanic['fare'], method='pearson'))

sns.pairplot(data=titanic, hue='survived')
plt.show()

'''
supervised learning
y = wx^T + b
weight와 곱해주는 값을 수치로 변환 -> ex) male = 1/2, female = 1/2
'''
