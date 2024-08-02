import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(filepath_or_buffer='pima-indians-diabetes3.csv')
print(df.head(10))
print(df['diabetes'].value_counts())
print(df.describe())

# Heat map
print(df.corr()) # 상관관계 출력
color_map = plt.cm.gist_heat # 그래프의 상관관계
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=color_map, linecolor='white', annot=True)
plt.show()

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

df = pd.read_csv(filepath_or_buffer='pima-indians-diabetes3.csv')
X = df.iloc[:,:8]
y = df.iloc[:,8] # 정답

model = Sequential(name='PIMA_INDIANS')
model.add(Input(shape=(8,)))
model.add(Dense(units=16, activation='relu', name='LAYER1'))
model.add(Dense(units=8, activation='relu', name='LAYER2'))
model.add(Dense(units=4, activation='relu', name='LAYER3'))
model.add(Dense(units=1, activation='sigmoid', name='OUTPUT'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.1))
model.fit(X, y, epochs=500, batch_size=10)

y_predict = model.predict(X)
print(y_predict)
print(y)

plt.plot(y_predict, y)
plt.show()