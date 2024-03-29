import numpy as np
import matplotlib.pyplot as plt
import math


def computeCost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost
    return total_cost


def computeGradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


def gradientDescent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    j_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 100000:
            j_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, j_history, p_history


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
m = x_train.shape[0]
w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2
w_final, b_final, j_hist, p_hist = gradientDescent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, computeCost,
                                                   computeGradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

w = 200
b = 100

# tmp_f_wb = computeModelOutput(x_train, w, b)
# plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
# plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
# plt.legend()
# plt.show()