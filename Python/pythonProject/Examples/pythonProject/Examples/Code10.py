from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
print(digits)
print(digits.images[0])
print(digits.data[0].shape)
plt.imshow(digits.images[2], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
plt.imshow(digits.data[0].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# n_samples = len(digits.images)
# print(n_samples)
# data = digits.images.reshape((1791, -1))

(X_train, X_test, y_train, y_test) = train_test_split(np.array(digits.data),
                                                      digits.target, test_size=0.3,
                                                      random_state=42)
k = 5
knn = KNeighborsClassifier(k)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
y_predictions = knn.predict(X_test)
print(y_predictions)
print(y_test)

print(metrics.accuracy_score(y_test, y_predictions))

print(y_test[0])
plt.imshow(X_test[0].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

y_predict = knn.predict([X_test[0]])
print(y_predict)
print(y_test[0])