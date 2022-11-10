#k=1000时求数值解
import numpy as np
L = np.array([[0.6, 0.1, 0.3], [0.1, 0.9, 0], [0.3, 0, 0.7]])
x = np.array([2, 1, 1]).T

k = 1000
Lk = L
for i in range(k):
    Lk = Lk.dot(L)

print(Lk.dot(x))                # 输出数值解