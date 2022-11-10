from matplotlib import pyplot as plt
import cvxpy as cp
from math import acos

x = []
y = []
z = []
with open('./points_xyz_162.txt') as f:
    lines = f.readlines()
for line in lines:
    xi, yi, zi = line.split(' ')
    x.append(float(xi))
    y.append(float(yi))
    z.append(float(zi))

fig = plt.figure()
ax1 = plt.axes(projection='3d')
#
# ax1.scatter3D(x, y, z, c='b')
# plt.show()

h = cp.Variable(len(x), boolean=True)
theta = cp.Variable(1)
obj = cp.Maximize(theta)
constraints = []

# print(x)
# print(x[1])

for i in range(len(x)):
    for j in range(i + 1, len(x)):
        if abs(x[i] * x[j] + y[i] * y[j] + z[i] * z[j]) >= 0.9999:
            continue
        constraints += [
            theta - (1 - h[i]) * 999 - (1 - h[j]) * 999 <= acos(x[i] * x[j] + y[i] * y[j] + z[i] * z[j])
        ]

constraints += [
    cp.sum(h, axis=0, keepdims=True) == 30
]

prob = cp.Problem(obj, constraints)
prob.solve(verbose=True)

print(h.value)
print(prob.value)
print(theta.value)
x_choose = []
y_choose = []
z_choose = []
print(h[0].value)
for i in range(162):
    if h[i].value:
        ax1.scatter3D(x[i], y[i], z[i], c='r')
    else:
        ax1.scatter3D(x[i], y[i], z[i], c='b')
# ax2 = plt.axes(projection='3d')
# print(x_choose)
# ax1.scatter3D(x_choose, y_choose, z_choose, c='r')
plt.show()
