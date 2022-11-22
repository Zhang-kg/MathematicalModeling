import math

import numpy as np
import pylab as plt
from matplotlib.pyplot import MultipleLocator
from scipy.optimize import curve_fit

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

def RMSE(y_pred, y_label):
    loss = 0.0
    for i in range(len(y_label)):
        loss += (y_label[i] - y_pred[i]) ** 2
    return np.sqrt(loss / len(y_label))


h = 0.001
N = 1000
M = 3
K_set = [i for i in range(1, 6)]
for K in K_set:
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
    x_test = x_shuffle[train_size + validate_size:]
    y_test = y_shuffle[train_size + validate_size:]


    def my_func(x, *arguments):
        y_pred = np.zeros(x.shape)
        for i in range(K):
            for j in range(M):
                y_pred += arguments[i * M + j] * np.power(x, j) * \
                          Gauss(x, arguments[K * M + i], arguments[K * M + K + i])
        return y_pred

    popt = curve_fit(my_func, x_train, y_train, p0=np.random.normal(size=K * M + 2 * K), maxfev=500000)[0]

    y_train_pred = my_func(x_train, *popt)
    y_test_pred = my_func(x_test, *popt)

    train_loss.append(RMSE(y_train_pred, y_train))
    test_loss.append(RMSE(y_test_pred, y_test))

# x_major_locator = MultipleLocator(2)
plt.subplot(1, 2, 1)
# ax = plt.gca()  # ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
plt.plot(K_set, train_loss)
# plt.xlim(0, K_set[-1] + 1)
plt.xlabel('K')
plt.ylabel('训练集上RMSE')
plt.title("prob1-3 K 训练集")

plt.subplot(1, 2, 2)
# ax = plt.gca()  # ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
plt.plot(K_set, test_loss)
# plt.xlim(0, K_set[-1] + 1)
plt.xlabel('K')
plt.ylabel('测试集上RMSE')
plt.title("prob1-3 K 测试集")

plt.show()
