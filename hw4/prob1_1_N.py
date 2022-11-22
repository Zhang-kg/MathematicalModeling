import math

import numpy as np
from scipy.optimize import curve_fit, least_squares, leastsq
import pylab as plt

np.random.seed(2)  # 便于复现

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.figure(figsize=(12, 6), dpi=100)

train_loss = []
test_loss = []

train_rate = 0.8
valid_rate = 0.1
test_rate = 0.1


def Gauss(x, a, b):
    return 1 / (b * np.sqrt(2 * math.pi)) * \
           np.exp(-((x - a) ** 2) / (2 * b ** 2))


def my_func(x, *args):
    return args[0] * Gauss(x, args[1], args[2]) + \
           args[3] * Gauss(x, args[4], args[5])


def RMSE(y_pred, y_label):
    loss = 0.0
    for i in range(len(y_label)):
        loss += (y_label[i] - y_pred[i]) ** 2
    return np.sqrt(loss / len(y_label))


h = 0.001
N_set = [i for i in range(100, 1001)]
for N in N_set:
    x = np.linspace(0, 4, N)
    y_withoutNoise = Gauss(x, 0, 1) + Gauss(x, 1.5, 1)
    y = y_withoutNoise + np.random.normal(scale=np.sqrt(h), size=y_withoutNoise.shape)

    index = [i for i in range(len(y))]
    np.random.shuffle(index)
    x_shuffle = x[index]
    y_shuffle = y[index]
    train_size = int(len(x_shuffle) * train_rate)
    validate_size = int(len(x_shuffle) * valid_rate)
    test_size = int(len(x_shuffle) * test_rate)

    x_train = x_shuffle[0: train_size]
    y_train = y_shuffle[0: train_size]
    x_validate = x_shuffle[train_size: train_size + validate_size]
    y_validate = y_shuffle[train_size: train_size + validate_size]
    x_test = x_shuffle[train_size + validate_size: ]
    y_test = y_shuffle[train_size + validate_size: ]

    popt = curve_fit(my_func, x_train, y_train, p0=np.ones(6), maxfev=500000)[0]

    y_train_pred = my_func(x_train, *popt)
    y_test_pred = my_func(x_test, *popt)

    train_loss.append(RMSE(y_train_pred, y_train))
    test_loss.append(RMSE(y_test_pred, y_test))

plt.subplot(1, 2, 1)
plt.plot(N_set, train_loss)
plt.xlabel('N')
plt.ylabel('训练集上RMSE')
plt.title("prob1-1 N 训练集 h = 0.001")

plt.subplot(1, 2, 2)
plt.plot(N_set, test_loss)
plt.xlabel('N')
plt.ylabel('测试集上RMSE')
plt.title("prob1-1 N 测试集 h = 0.001")

plt.show()