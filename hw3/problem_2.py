# coding=GBK

import cvxpy as cp
import numpy as np
import gurobipy


gurobipy.setParam("timeLimit", 600)  # �����10����
np.set_printoptions(threshold=np.Inf)

# �����
print("�����:")
# inputs
K = list(map(int, input("ѡȡ����Ŀ K: ").split()))
dataset = int(input("ѡȡ���ݼ�, 0 ���� 42 ����, 1 ���� 162 ����: "))
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
H = cp.Variable((len(K), len(dots)), integer=True)  # H[i][j]�����i��ѡ��j����
Dm = cp.Variable(len(K))  # Dm[i]�����i�����С���Ǿ���
dm = cp.Variable(1)  # dm���������С���Ǿ���
M = int(5)
cons = [Dm >= 0, Dm <= np.pi, dm >= 0, dm <= np.pi, H >= 0, H <= 1,
        np.ones(shape=(1, len(K)), dtype=int) @ H <= 1]  # һ���㲻�ܱ����ѡ��
for i in range(len(dots)):
    for j in range(i + 1, len(dots)):
        cons.append(dm <=  # ��֤dm���������С���Ǿ���
                    D[i, j] + M * (2 - cp.sum(H[..., i]) - cp.sum(H[..., j])))
for i in range(len(K)):
    cons.append(cp.sum(H[i]) == K[i])  # ÿ���鶼��ѡ����
    for j in range(len(dots)):
        for k in range(j + 1, len(dots)):
            cons.append(Dm[i] <=  # ��֤Dm[i]�ǵ�i�����С���Ǿ���
                        D[j, k] + M * (2 - H[i][j] - H[i][k]))

prob = cp.Problem(cp.Maximize(cp.sum(Dm) + dm),  # �Ż�Ŀ��Ϊ���� + ������С���Ǿ������
                  constraints=cons)
prob.solve(solver="GUROBI", verbose=True)

# ��
print("��󸲸Ǿ���Ϊ(rad):")
print("����: %g" % dm.value)
for i in range(len(K)):
    print("��%d��: %g" % (i + 1, Dm[i].value))
print("ѡȡ�ĵ�Ϊ:")
for i in range(len(K)):
    print("��%d��: " % (i + 1), end="")
    for j in range(len(dots)):
        print("%d" % (H[i][j].value + 1e-5), end=" ")
    print()
