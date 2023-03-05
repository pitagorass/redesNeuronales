import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0, 0.0, 2.0, 3.0, 4.0, -4.0], dtype=float)
ys = np.array([1.0, -2.0, 4.0, 6.0, 10.0, -14], dtype=float)

model.fit(xs, ys, epochs=1000)
print(model.predict([16.0]))

""" 
y = F(x) = 3x- 5
x | y
1  -2
2   1
n.. 
"""



