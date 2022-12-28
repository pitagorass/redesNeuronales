""" https://www.wolframalpha.com/ """
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(a):
    return 1/(1+np.exp(-a))


def step(x):
    return np.piecewise(x, [x < 0.0, x > 0.0], [0, 1])


def signum(s):
    return np.piecewise(s, [s >= 0.0, s < 0], [1, -1])


def relu(i):
    return np.piecewise(i, [i < 0, i > 0], [0, lambda i: i])


def tanh(a):
    return np.tanh(a)


def softmax(a):
    expo = np.exp(x)
    expo_sum = np.sum(np.exp(x))
    return expo/expo_sum


x = np.linspace(10, -10, 100)

plt.plot(x, sigmoid(x))
plt.show()

plt.plot(x, step(x))
plt.show()

plt.plot(x, signum(x))
plt.show()

plt.plot(x, relu(x))
plt.show()

plt.plot(x, tanh(x))
plt.show()

plt.plot(x, softmax(x))
plt.show()