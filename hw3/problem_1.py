# coding=GBK

import cvxpy as cp
import numpy as np
import gurobipy


gurobipy.setParam("timeLimit", 600)  # 最多跑3分钟
np.set_printoptions(threshold=np.Inf)

# 问题一
print("问题一:")
# inputs
k = int(input("选取点数目 K: "))
dataset = int(input("选取数据集, 0 代表 42 个点, 1 代表 162 个点: "))
path = "./points_xyz_42.txt" if dataset == 0 else "./points_xyz_162.txt" if dataset == 1 else None
dots = []
with open(path, "r") as f:
    lines = f.readlines()
    for line in lines:
        dots.append(list(map(np.float32, line.split())))
dots = np.asarray(dots, dtype=np.float32)

# calculate distance matrix
D = np.zeros(shape=(len(dots), len(dots)), dtype=np.float32)
for i in range(len(dots)):
    for j in range(i + 1, len(dots)):
        s = np.dot(dots[i], dots[j])
        if s < -1:
            s = -1
        if s > 1:
            s = 1
        D[i, j] = np.arccos(s)

# define problem
H = cp.Variable(len(dots), integer=True)
dm = cp.Variable(1)
M = int(5)
cons = [cp.sum(H) == k, dm >= 0, dm <= np.pi, H >= 0, H <= 1]
for i in range(len(dots)):
    for j in range(i + 1, len(dots)):
        cons.append(dm <= D[i, j] + M * (2 - H[i] - H[j]))

prob = cp.Problem(cp.Maximize(dm), constraints=cons)
prob.solve(solver="GUROBI", verbose=True)

# 答案
print("最大覆盖距离为(rad):")
print("%g" % dm.value)
print("选取的点为:")
for i in range(len(dots)):
    print("%d" % (H[i].value + 1e-6), end=" ")
