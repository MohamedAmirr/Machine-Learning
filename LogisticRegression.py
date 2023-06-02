import numpy as np
from random import random, seed
import matplotlib.pyplot as plt


def range1(arr, beg: int, end: int, dim):
    arr1 = np.array([])
    for i in range(beg, end):
        u = arr[i][dim]
        arr1 = np.append(arr1, u)
    return arr1


def sigmoid(z):
    return 1 / (1 + np.exp(z))


mean1 = [2, 2]
cov = [[1, 0],
       [0, 1]]
mean2 = [-2, 6]

datax1 = np.random.multivariate_normal(mean1, cov, 100)
datax2 = np.random.multivariate_normal(mean2, cov, 100)

data = np.concatenate((datax1, datax2))

y = np.concatenate((-1 * np.ones(shape=100, dtype=int),
                    np.ones(shape=100, dtype=int)))

x = (data - np.mean(data)) / np.std(data)  # standardization

plt.scatter(range1(data, 0, 100, 0), range1(data, 0, 100, 1))
plt.scatter(range1(data, 100, 200, 0), range1(data, 100, 200, 1))

# plt.show()

seed(1)
w = np.array([random(), random()])
b = random()
eta = 0.01
iterations = 100000

for i in range(iterations):
    z = np.dot(np.transpose(w), np.transpose(x)) + b
    phiZ = sigmoid(-z)

    w = w - (eta * np.dot(phiZ - y, x) / len(x))
    b = b - eta * np.mean(phiZ - y)

xx = np.array([-5,5])
yy = np.array([])
for i in xx:
    yy = np.append(yy, (-w[0] * i - b) / w[1])

plt.plot(xx, yy)
plt.show()
