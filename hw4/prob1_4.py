import math

import numpy as np
from scipy.optimize import curve_fit, least_squares, leastsq
from matplotlib import pyplot as plt

np.random.seed(1)  # 便于复现

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.figure(figsize=(12, 6), dpi=100)

def Gauss(x, a, b):
    return 1 / (b * np.sqrt(2 * math.pi)) * \
           np.exp(-((x - a) ** 2) / (2 * b ** 2))

def my_func(x, *arguments):
    a = arguments[0]
    b = arguments[1]
    c = arguments[2]
    y_pred = a * np.cos(b * x) + c
    return y_pred

def RMSE(y_pred, y_label):
    loss = 0.0
    for i in range(len(y_label)):
        loss += (y_label[i] - y_pred[i])**2
    return np.sqrt(loss / len(y_label))


N = 1000
h = 0.0002
train_rate = 0.8
validate_rate = 0.1
test_rate = 0.1
x = np.linspace(0, 4, N)
y_withoutNoise = Gauss(x, 0, 1) + Gauss(x, 1.5, 1)
y = y_withoutNoise + np.random.normal(scale=np.sqrt(h), size=y_withoutNoise.shape)

index = [i for i in range(len(y))]
np.random.shuffle(index)
x_shuffle = x[index]
y_shuffle = y[index]

# x_shuffle = x
# y_shuffle = y

train_size = int(len(x_shuffle) * train_rate)
validate_size = int(len(x_shuffle) * validate_rate)
test_size = int(len(x_shuffle) * test_rate)
x_train = x_shuffle[0: train_size]
y_train = y_shuffle[0: train_size]
x_validate = x_shuffle[train_size: train_size + validate_size]
y_validate = y_shuffle[train_size: train_size + validate_size]
x_test = x_shuffle[train_size + validate_size: train_size + validate_size + test_size]
y_test = y_shuffle[train_size + validate_size: train_size + validate_size + test_size]

print(len(x_train))
print(len(x_validate))
print(len(x_test))


popt = curve_fit(my_func, x_train, y_train, maxfev=500000, p0=np.random.normal(size=3))[0]
print("参数为" + str(popt))
y_train_pred = my_func(x_train, *popt)
print("训练集上的RMSE = " + str(RMSE(y_train_pred, y_train)))
print()
y_validate_pred = my_func(x_validate, *popt)
print("验证集上的RMSE = " + str(RMSE(y_validate_pred, y_validate)))
y_test_pred = my_func(x_test, *popt)
print("测试集上的RMSE = " + str(RMSE(y_test_pred, y_test)))

plt.subplot(1, 2, 1)
plt.plot(x, y_withoutNoise, linestyle='--', c='gold')
plt.scatter(x, y)
plt.scatter(x_train, y_train_pred, c='r')
# plt.scatter(x_test, y_test_pred, c='r')
plt.title("训练集")

plt.subplot(1, 2, 2)
plt.plot(x, y_withoutNoise, linestyle='--', c='gold')
plt.scatter(x, y)
# plt.scatter(x_train, y_train_pred, c='r')
plt.scatter(x_test, y_test_pred, c='r')
plt.title("测试集")
# plt.plot(x_train, y_train_pred)

plt.show()