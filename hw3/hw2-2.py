import sys

import cvxpy as cp
import numpy as np
import gurobipy as gp
from matplotlib import pyplot as plt
from math import acos

gp.setParam("TimeLimit", 180)
gp.setParam("MIPFocus", 3)

Ks = []
print('使用分组思路求解思考题')
pathBinary = int(input("选择数据集(0表示42点，1表示162点):"))
path = ""
if pathBinary == 0:
    path = "./points_xyz_42.txt"
elif pathBinary == 1:
    path = "./points_xyz_162.txt"
else:
    print("点集错误")
    sys.exit(1)
S = int(input("总组数: "))
for i in range(S):
    Ks.append(int(input("第" + str(i) + "组内点数: ")))
pointsSum = np.sum(Ks)
print("选取总点数为: " + str(pointsSum))
if pointsSum > (42 if pathBinary == 0 else 162):
    print("选取点数多余点集")
    sys.exit(1)
w = float(input("组内权重w(0~1之间的浮点数): "))

x = []
y = []
z = []

with open(path) as f:
    lines = f.readlines()
for line in lines:
    xi, yi, zi = line.split(' ')
    x.append(float(xi))
    y.append(float(yi))
    z.append(float(zi))

# h_Matrix 表示选择矩阵
h_Matrix = cp.Variable((len(x), S), boolean=True)
thetas = cp.Variable(S + 1)
obj = cp.Maximize(w * (cp.sum(thetas[0:S])) / S + (1 - w) * thetas[S])
constraints = [thetas[S] >= 0.70]

# theta 的约束
for s in range(S):
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if abs(x[i] * x[j] + y[i] * y[j] + z[i] * z[j]) > 1:
                continue
            constraints += [
                thetas[s] - (1 - h_Matrix[i][s]) * 999 - (1 - h_Matrix[j][s]) * 999 <=
                acos(x[i] * x[j] + y[i] * y[j] + z[i] * z[j])
            ]

for i in range(len(x)):
    for j in range(i + 1, len(x)):
        if abs(x[i] * x[j] + y[i] * y[j] + z[i] * z[j]) > 1:
            continue
        constraints += [
            thetas[S] - (1 - cp.sum(h_Matrix[i, :])) * 999 - (1 - cp.sum(h_Matrix[j, :])) * 999 <=
            acos(x[i] * x[j] + y[i] * y[j] + z[i] * z[j])
        ]

# 点只能选择一次
for i in range(len(x)):
    constraints += [
        cp.sum(h_Matrix[i, :]) <= 1
    ]

# 每组个数
for s in range(S):
    constraints += [
        cp.sum(h_Matrix[:, s]) == Ks[s]
    ]

prob = cp.Problem(obj, constraints)
prob.solve(reoptimize=True, solver="GUROBI", verbose=True)

print(h_Matrix.value)
print(thetas.value)
print(obj.value)

fig = plt.figure()
ax1 = plt.axes(projection='3d')
cArray = ['r', 'blue', 'orange', 'gold', 'yellow', 'purple',
          'darksage', 'green', 'cyan', 'saddlebrown', 'greenyellow', 'deeppink']
for i in range(len(x)):
    for j in range(len(Ks)):
        if h_Matrix[i][j].value:
            ax1.scatter3D(x[i], y[i], z[i], c=cArray[j])
    if not np.sum(h_Matrix[i, :].value):
        ax1.scatter3D(x[i], y[i], z[i], c='black')

plt.show()
