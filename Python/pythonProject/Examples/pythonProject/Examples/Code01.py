import numpy as np
from matplotlib import pyplot as plt

a:np.ndarray = np.array(object=[1, 2, 3]) # (:) 파스칼 노테이션
print(a)
print(type(a))
b = np.array([1, 2, 3, 4, 5, 6, 7.0, 8, 9, 10])
print(b)
print(type(b))

c = np.zeros(shape=(3, 4)) # 행렬 초기화
print(f'NP.ZEROS : \n{c}')

print(a[0])
print(a[1])
print(c[0][0])
print(c[0, 0])
d = np.ones((4, 4))
print(d)

result = np.arange(start=0, stop=6, step=1)
print(result)
print(np.linspace(0, 10, 10))

np_array = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print(np_array)
print(np.sort(np_array))
print(np_array)
np_array_sorting = np.sort(np_array)
print(np_array_sorting)

x1 = np.array([[1, 2, ], [3, 4, ]]) # tray comma?
y1 = np.array([[5, 6, ], [7, 8, ]])
result1 = np.concatenate((x1, y1), axis=0)
print(result1)

x2 = np.array([[1, 2, ], [3, 4, ]]) # tray comma?
y2 = np.array([[5, 6, ], [7, 8, ]])
result2 = np.concatenate((x2, y2), axis=1)
print(result2)

array_1 = np.arange(12)
print(array_1)
print(array_1.reshape(3, 4))
print(array_1.reshape(4, 3))

# 차원 증가, 차원 축소
array_5 = np.array([1, 2, 3, 4, 5, 6])
print(f'ARRAY_5 : {array_5.shape}') # (6, ) -> 벡터
array_5_1 = array_5[np.newaxis, :] # (:) 반복 구분
print(array_5_1)
print(array_5_1.shape) # shape은 numpy계열만 사용가능

array_1_5 = array_5[:, np.newaxis]
print(array_1_5)
print(array_1_5.shape)

ages = np.array([18, 19, 25, 30, 28,])
print(ages[0]) # 단일 값
print(ages[:])
print(ages[1:]) # 차원이 줄지 않음 []

# 복수의 값이 있으면 변수s
scores = np.array([[99, 93, 60, ], [98, 82, 93, ], [93, 65, 81, ], [78, 82, 81, ]])
print(scores.mean(axis=0))
print(scores.mean(axis=1))

x = np.arange(start=0, stop=20, step=1) # 독립변수 : 벡터 x가 가지는 범위(축)
y1 = 2 * np.ones(20)
y2 = x
y3 = np.square(x)
plt.plot(x, y1, x, y2, x, y3)
plt.show()

'''
Hyper plane : 4차원 평면
w1x1 + w2x2 + w3x3 + w4x4 + ... +b(bias)
CNN : 2차원, 3차원
flatten() : 원본 배열 수정 안됨
f(x) = 1 : 상수함수 Linear
f(x) = x : 선형함수 Linear
f(x) = x^2 : 비선형 nonlinear
'''