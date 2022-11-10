# coding=GBK

import cvxpy as cp
import numpy as np
import gurobipy


gurobipy.setParam("timeLimit", 600)  # 最多跑10分钟
np.set_printoptions(threshold=np.Inf)

# 问题二
print("问题二:")
# inputs
K = list(map(int, input("选取点数目 K: ").split()))
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
H = cp.Variable((len(K), len(dots)), integer=True)  # H[i][j]代表第i组选第j个点
Dm = cp.Variable(len(K))  # Dm[i]代表第i组的最小覆盖距离
dm = cp.Variable(1)  # dm是总体的最小覆盖距离
M = int(5)
cons = [Dm >= 0, Dm <= np.pi, dm >= 0, dm <= np.pi, H >= 0, H <= 1,
        np.ones(shape=(1, len(K)), dtype=int) @ H <= 1]  # 一个点不能被多次选择
for i in range(len(dots)):
    for j in range(i + 1, len(dots)):
        cons.append(dm <=  # 保证dm是总体的最小覆盖距离
                    D[i, j] + M * (2 - cp.sum(H[..., i]) - cp.sum(H[..., j])))
for i in range(len(K)):
    cons.append(cp.sum(H[i]) == K[i])  # 每个组都能选够点
    for j in range(len(dots)):
        for k in range(j + 1, len(dots)):
            cons.append(Dm[i] <=  # 保证Dm[i]是第i组的最小覆盖距离
                        D[j, k] + M * (2 - H[i][j] - H[i][k]))

prob = cp.Problem(cp.Maximize(cp.sum(Dm) + dm),  # 优化目标为各组 + 总体最小覆盖距离最大
                  constraints=cons)
prob.solve(solver="GUROBI", verbose=True)

# 答案
print("最大覆盖距离为(rad):")
print("总体: %g" % dm.value)
for i in range(len(K)):
    print("第%d组: %g" % (i + 1, Dm[i].value))
print("选取的点为:")
for i in range(len(K)):
    print("第%d组: " % (i + 1), end="")
    for j in range(len(dots)):
        print("%d" % (H[i][j].value + 1e-5), end=" ")
    print()
