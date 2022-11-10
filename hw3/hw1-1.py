import cvxpy as cp
from scipy.optimize import linprog
from numpy import array

c = array([20, 90, 80, 70, 30])
a = array([
    [-1, -1, 0, 0, -1],
    [0, 0, -1, -1, 0],
    [3, 0, 2, 0, 0],
    [0, 3, 0, 2, 1]
])
b = array([-30.5, -30, 120, 48])
x = cp.Variable(5, pos=True)
obj = cp.Minimize(c @ x)
cons = [a @ x <= b]
prob = cp.Problem(obj, cons)
prob.solve()

print("最优解为：", x.value)
print("最优值为：", prob.value)
print("a @ x = ", a@x.value)