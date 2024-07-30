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

# 오류 could not convert string to float -> string 값 제외
titanic2 = titanic.select_dtypes(include=[int, float, bool])
print(titanic2)
print(titanic2.shape)

titanic_corr = titanic2.corr(method='pearson')
print(titanic_corr)
print(titanic_corr.shape)

# 엑셀에서 읽을 수 있도록 CSV 변환
titanic_corr.to_csv('Titanic_corr.csv', index=False)

# 생존했는데, 성인남자인 경우의 확률
print(f"성인남자의 생존률 : {titanic['survived'].corr(titanic['adult_male'])}")
print(f"운임요금의 생존률 : {titanic['survived'].corr(titanic['fare'])}")

# 산점도 (scatter)
sns.pairplot(data=titanic, hue='survived')
plt.show()

# heat_map : 열로써 상관관계를 봄, 빨간색일수록 서로 연관관계가 크다.
# 나잇 대 문제가 발생 10, 20, 30, 40, 50, 60, 70 -> 값이 커서 bias가 매우 커짐
def category_age(x:int) -> int: # 파라미터 자료형 명시, 반환 자료형 명시
    if   x < 10: return 0
    elif x < 20: return 1
    elif x < 30: return 2
    elif x < 40: return 3
    elif x < 50: return 4
    elif x < 60: return 5
    elif x < 70: return 6
    else:        return 7

# age를 가져와서 나이에 따른 category_age반환 값을 할당
titanic['age2'] = titanic['age'].apply(category_age) # 콜백 함수

# 성별을 숫자형태로 변환
titanic['sex'] = titanic['sex'].map({'male':1, 'female':0})
# sibsp: 동반한 형재자매와 배우자 수, parch : 함게 탑승한 부모, 자녀 수 총합, [+ 1:자기 자신]
titanic['family'] = titanic['sibsp'] + titanic['parch'] + 1
titanic.to_csv('NewTitanic.csv', index=False)

heat_map = titanic[['survived', 'sex', 'age2', 'family', 'pclass', 'fare']]
color_map = plt.cm.RdBu
sns.heatmap(heat_map.astype(float).corr(), linewidths=0.2,
            vmax=1.0, square=True, cmap=color_map, linecolor='white',
            annot=True, annot_kws={'size': 8})
plt.show()

# titanic_correlation_analysis = titanic2.corr(method='pearson')

# print(f'Pearson correlation : \r\n{titanic_correlation_analysis}')
# print(titanic['survived'].corr(other=titanic['adult_male'], method='pearson'))
# print(titanic['survived'].corr(other=titanic['fare'], method='pearson'))

'''
supervised learning
y = wx^T + b
weight와 곱해주는 값을 수치로 변환 -> ex) male = 1/2, female = 1/2
퍼셉트론-> 역전파알고리즘-> svm-> 필기체 인식->  ImageNet에서 우승
y - y^ = 0, 차이, loss가 0이 되도록!
overfit : 67p 그래프, 부드러운 그래프로 
클러스터링 : 같은 집단끼리 뭉쳐져있는

'''