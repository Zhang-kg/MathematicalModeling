import numpy as np
from scipy.optimize import minimize

K = []
S = int(input("组数："))
for i in range(S):
    K.append(int(input("第" + str(i + 1) + "组点数：")))

w = float(input("分组权重："))

print(K)


def obj(x):
    return -(w * np.sum(x[int(np.sum(K)) * 3: int(np.sum(K)) * 3 + S]) / S +
             (1 - w) * x[int(np.sum(K)) * 3 + S])


def constr1(x):
    cons = []
    for i in range(int(np.sum(K))):
        cons.append(x[3 * i] ** 2 + x[3 * i + 1] ** 2 + x[3 * i + 2] ** 2 - 1)
    return cons


def constr2(x):
    cons = []
    for s in range(S):
        pointsNum = K[s]
        theta = x[int(np.sum(K)) * 3 + s]
        base = int(np.sum(K[: s])) * 3
        for i in range(pointsNum):
            for j in range(i + 1, pointsNum):
                cons.append(
                    np.arccos((x[3 * i + base] * x[3 * j + base] +
                               x[3 * i + 1 + base] * x[3 * j + 1 + base] +
                               x[3 * i + 2 + base] * x[3 * j + 2 + base]) /
                              np.sqrt((x[3 * i + base] ** 2 + x[3 * i + 1 + base] ** 2 + x[3 * i + 2 + base] ** 2) *
                                      (x[3 * j + base] ** 2 + x[3 * j + 1 + base] ** 2 + x[3 * j + 2 + base] ** 2))) - theta)
    return cons


def constr3(x):
    cons = []
    theta = x[int(np.sum(K)) * 3 + S]
    for i in range(int(np.sum(K))):
        for j in range(i + 1, int(np.sum(K))):
            cons.append(np.arccos((x[3 * i] * x[3 * j] +
                                   x[3 * i + 1] * x[3 * j + 1] +
                                   x[3 * i + 2] * x[3 * j + 2]) /
                                  np.sqrt((x[3 * i] ** 2 + x[3 * i + 1] ** 2 + x[3 * i + 2] ** 2) *
                                          (x[3 * j] ** 2 + x[3 * j + 1] ** 2 + x[3 * j + 2] ** 2))) - theta)
    return cons

con1 = {'type': 'eq', 'fun': constr1}
con2 = {'type': 'ineq', 'fun': constr2}
con3 = {'type': 'ineq', 'fun': constr3}

res = minimize(fun=obj, x0=np.random.normal(size=3 * int(np.sum(K)) + S + 1), constraints=[con1, con2, con3], method="SLSQP")

print(res)
