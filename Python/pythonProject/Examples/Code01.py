import numpy as np
a:np.ndarray = np.array(object=[1, 2, 3]) # (:) 파스칼 노테이션
print(a)
print(type(a))
b = np.array([1,2,3,4,5,6,7.0,8,9,10])
print(b)
print(type(b))

c = np.zeros(shape=(3,4)) # 행렬 초기화
print(f'NP.ZEROS : \n{c}')