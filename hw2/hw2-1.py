import matplotlib.pyplot as plt
import numpy as np
# 数值差分解
x = [-2, 0]
for i in range(2, 2000):
    x.append(0.5 * x[i - 1] + x[i - 2])
n = range(0, 2000)
c1 = 'r'
plt.plot(n, x, c1, linewidth=1, marker='d')

# 解析解
x2 = (np.sqrt(17)-17)/17*((1+np.sqrt(17))/4)**n-(17+np.sqrt(17))/17*((1-np.sqrt(17))/4)**n
c2 = 'b'
plt.plot(n, x2, 'b')
plt.show()