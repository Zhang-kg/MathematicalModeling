import sympy as sp
from sympy import Eq, Derivative

x = sp.symbols('x', cls=sp.Function)
y = sp.symbols('y', cls=sp.Function)
t = sp.symbols('t')

eq = (Eq(Derivative(x(t), t, 1), x(t) - 2 * y(t)), Eq(Derivative(y(t), t, 1), x(t) + 2 * y(t)))
result = sp.dsolve(eq, ics={y(0): 0, x(0): 1})
print(result)