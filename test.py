import tensorflow as tf
import numpy as np

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 넘파이 데이터를 텐서 데이터로 변환
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_train = tf.one_hot(y_train, depth=len(np.unique(y_train)))
y_test = tf.one_hot(y_test, depth=len(np.unique(y_train)))

# 레이어 설계
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])

# CPU 학습
print("CPU를 사용한 학습")
with tf.device("/device:CPU:0"):
    model.fit(x_train, y_train, batch_size=32, epochs=3)

print("GPU를 사용한 학습")
with tf.device("/device:GPU:0"):
    model.fit(x_train, y_train, batch_size=32, epochs=3)