import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

train_images=mnist.train_images(target_dir="/Users/qs/PyCharmMiscProject/KerasForNeuralNetwork/data/mnist/MNIST/raw")
train_labels=mnist.train_labels(target_dir="/Users/qs/PyCharmMiscProject/KerasForNeuralNetwork/data/mnist/MNIST/raw")
test_images=mnist.test_images(target_dir="/Users/qs/PyCharmMiscProject/KerasForNeuralNetwork/data/mnist/MNIST/raw")
test_labels=mnist.test_labels(target_dir="/Users/qs/PyCharmMiscProject/KerasForNeuralNetwork/data/mnist/MNIST/raw")

train_images=(train_images/255)-0.5
test_images=(test_images/255)-0.5

train_images=train_images.reshape((-1,784))
test_images=test_images.reshape((-1,784))

model = Sequential([
    Dense(64,activation='relu',input_shape=(784,)),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax'),
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32,
)