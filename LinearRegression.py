from random import random
import matplotlib.pyplot as plt
import numpy as np

x = np.array([i for i in range(10)])
y = np.array([i for i in range(10)])

w = random()
b = random()

plt.scatter(x, y)

for i in range(100000):
    yHat = w * x + b
    dJdW = np.dot((.001 * (1 / len(x)) * (yHat - y)), np.transpose(x))
    dJdB = .001 * np.mean(yHat - y)

    w = w - dJdW
    b = b - dJdB

newX = [i for i in range(-10, 10)]
newY = [w * newX[i] + b for i in range(20)]

plt.plot(newX, newY)
plt.show()
