from random import seed, random

import numpy as np
import matplotlib.pyplot as plt


def range1(arr, beg: int, end: int, dim):
    arr1 = np.array([])
    for i in range(beg, end):
        u = arr[i][dim]
        arr1 = np.append(arr1, u)
    return arr1


mean1 = [2, 2]
cov = [[1, 0],
       [0, 1]]
mean2 = [-2, 6]

datax1 = np.random.multivariate_normal(mean1, cov, 100)
datax2 = np.random.multivariate_normal(mean2, cov, 100)

data = np.concatenate((datax1, datax2))

labels = np.concatenate((-1 * np.ones(shape=100, dtype=int),
                         np.ones(shape=100, dtype=int)))

data = (data - np.mean(data)) / np.std(data)  # standardization

plt.scatter(range1(data, 0, 100, 0), range1(data, 0, 100, 1))
plt.scatter(range1(data, 100, 200, 0), range1(data, 100, 200, 1))
# plt.show()

seed(1)
w = np.array([random(), random()])
b = random()
eta = 0.01

iter = 10000

for i in range(iter):
    error = 0
    for j in range(len(labels)):
        x = data[j]
        y = labels[j]

        z = np.dot(np.transpose(w), x) + b
        if z >= 0:
            z = 1
        else:
            z = -1

        w = w - eta * (z - y) * x
        b = b - eta * (z - y)
        error += abs(y - z)
    if error == 0:
        break

x = np.array([i for i in range(-5, 5)])
y = np.array([])
for i in x:
    y = np.append(y, (-w[0] * i - b) / w[1])

plt.plot(x, y)
plt.show()
