import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import Eq, Derivative

x = sp.symbols('x', cls=sp.Function)
y = sp.symbols('y', cls=sp.Function)
t = sp.symbols('t')

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)
plt.rc('font', size=16)

eq = (Eq(Derivative(x(t), t, 1), x(t) - 2 * y(t)), Eq(Derivative(y(t), t, 1), x(t) + 2 * y(t)))
result = sp.dsolve(eq, ics={y(0): 0, x(0): 1})
result1 = sp.lambdify(t, result[0].args[1], 'numpy')
result2 = sp.lambdify(t, result[1].args[1], 'numpy')
tt = np.linspace(0, 1, 101)
plt.plot(result1(tt), result2(tt), 'b')
plt.xlabel('x')
plt.ylabel('y') # x, y 轴添加标签

x_major_locator = plt.MultipleLocator(0.25)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

# plt.plot(tt, result2(tt), 'r')
plt.show()