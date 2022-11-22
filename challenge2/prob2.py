import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

fig = plt.figure()
ax1 = plt.axes(projection='3d')

K = 42


def obj(x):
    theta = x[3 * K]
    return -theta


def constr1(x):
    theta = x[3 * K]
    cons = []
    for i in range(K):
        for j in range(i + 1, K):
            cons.append(np.arccos((x[3 * i] * x[3 * j] +
                                   x[3 * i + 1] * x[3 * j + 1] +
                                   x[3 * i + 2] * x[3 * j + 2]) /
                                  np.sqrt((x[3 * i] ** 2 + x[3 * i + 1] ** 2 + x[3 * i + 2] ** 2) *
                                          (x[3 * j] ** 2 + x[3 * j + 1] ** 2 + x[3 * j + 2] ** 2))) - theta)
    return cons


def constr2(x):
    cons = []
    for i in range(K):
        cons.append(x[3 * i] ** 2 + x[3 * i + 1] ** 2 + x[3 * i + 2] ** 2 - 1)
    return cons


con1 = {'type': 'ineq', 'fun': constr1}
con2 = {'type': 'eq', 'fun': constr2}

res = minimize(fun=obj, x0=np.random.normal(size=3 * K + 1), constraints=[con1, con2], method="SLSQP")

print(res)
for i in range(K):
    ax1.scatter3D(res.x[3 * i], res.x[3 * i + 1], res.x[3 * i + 2], c='r')
plt.show()