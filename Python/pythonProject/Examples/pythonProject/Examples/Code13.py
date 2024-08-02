import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x:float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

X = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x=X)
plt.plot(X,y)
plt.show()

# x = np.linspace(-np.pi, np.pi, 60)
# y = np.tanh(x)
# plt.plot(x,y)
# plt.show()

# def relu(x:float) -> int:
#     return np.maximum(x, 0) # 0보다 크면, 자기 자신
# x = np.arange(-10.0, 10.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# plt.show()