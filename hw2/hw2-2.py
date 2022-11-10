import sympy as sp
import numpy as np
from sympy import Eq, Derivative
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.rc('font', family='SimHei')
plt.rc('axes', unicode_minus=False)
plt.rc('font', size=16)

x = sp.symbols('x', cls=sp.Function)
y = sp.symbols('y', cls=sp.Function)
t = sp.symbols('t')

eq = (Eq(Derivative(x(t), t, 1), x(t) - 2 * y(t)), Eq(Derivative(y(t), t, 1), x(t) + 2 * y(t)))
result = sp.dsolve(eq, ics={y(0): 0, x(0): 1})
result1 = sp.lambdify(t, result[0].args[1], 'numpy')
result2 = sp.lambdify(t, result[1].args[1], 'numpy')
tt = np.linspace(0, 1, 101)
plt.plot(result1(tt), result2(tt))
# plt.plot(tt, result2(tt), )
ttt = np.linspace(0, 1, 20)
df = lambda f, t2: [f[0] - 2 * f[1], f[0] + 2 * f[1]]
s1 = odeint(df, [1, 0], ttt)
print(s1[:, 0])
plt.plot(s1[:, 0], s1[:, 1], 'p')
plt.legend(['[0,1]上的解析解', '[0,1]上的数值解'])
plt.show()
