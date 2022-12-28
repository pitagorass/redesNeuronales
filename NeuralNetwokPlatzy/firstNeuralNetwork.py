import numpy as np
from keras import layers, models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data.shape
train_data[1]
plt.imshow(train_data[1])
train_labels[1]

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics='accuracy')
model.summary()
x_train = train_data.reshape((60000, 28*28))
x_train = x_train.astype('float32')/255

x_test = test_data.reshape((10000, 28*28))
x_test = x_test.astype('float32')/255

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

y_train[0]

model.fit(x_train, y_train, epochs=5, bach_size=128)
model.evaluate(x_test, y_test)
