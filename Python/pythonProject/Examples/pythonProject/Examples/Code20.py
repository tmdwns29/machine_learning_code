import tensorflow as tf
import keras

# VGG19_NETWORK 모델 생성
model = keras.models.Sequential(name='VGG19_NETWORK')

# 입력층
input = keras.Input(shape=(224, 224, 3))
model.add(input)

# block 1
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

# block 2
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

# block 3
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

# block 4
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

# blcok 5
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3),
                              padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

# DNN
model.add(keras.layers.Flatten()) # Fully connected
model.add(keras.layers.Dense(units=4096, activation='relu', name='LAYER1'))
model.add(keras.layers.Dense(units=4096, activation='relu', name='LAYER2'))
model.add(keras.layers.Dense(units=1000, activation='relu', name='OUTPUT'))
model.compile()
model.summary()
model.compile(optimizer='adam', loss_weights='categorical_crossentropy')
# --- 컴파일만 진행 ---

# 완성된 모델 가져오기
model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels.h5')
import cv2
image1 = cv2.imread(filename='laptop.jpg') # 이미지 읽기

# 이미지 
cv2.imshow(winname='CAT', mat=image1)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 새로운 이미지 재정의
new_image1 = cv2.resize(src=image1, dsize=(224, 224))
import numpy as np

new_image1 = new_image1[np.newaxis, :] # 행 추가
predict_image = model.predict(new_image1) # 새로운 이미지 예측
print(predict_image) # 예측 값
print(f'가장 높은 확률 이미지 : {np.argmax(predict_image)}') # 예측 값이 가장 높은 인덱스값(classes.txt)