#程序文件ex5_2.py
# %%
import cvxpy as cp
from scipy.optimize import linprog
from numpy import array

c = array([70, 50, 60])  #定义目标向量
# c = array([70, 50.0001, 60])  #定义目标向量
a = array([[2, 4, 3], [3, 1, 5], [7, 3, 5]])  #定义约束矩阵
b = array([150, 160, 200])  #定义约束条件的右边向量
x = cp.Variable(3, pos=True)  #定义3个决策变量
obj = cp.Maximize(c@x)    #构造目标函数
cons = [a@x <=b]     #构造约束条件
prob = cp.Problem(obj, cons)
prob.solve()   #求解问题
# prob.solve(solver='GUROBI')   #用gurobi求解器求解问题
print('最优解为：', x.value)
print('最优值为：', prob.value)
print('a*x=', a@x.value)

res = linprog(-c, A_ub=a, b_ub=b, bounds=[(0,None)]*3)
print('最优解为：', res.x)
print('最优值为：', res.fun)
print('a*x=', a@res.x)


