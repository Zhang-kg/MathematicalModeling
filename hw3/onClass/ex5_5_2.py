#程序文件data5_5_2.py
# %%
from xmlrpc.client import boolean
import cvxpy as cp
import pandas as pd

data=pd.read_excel("data5_5_3.xlsx", header=None)
data=data.values; c=data[:-1,:-1]
d=data[-1,:-1]; e=data[:-1,-1]

x=cp.Variable((6,8), pos=True)
obj=cp.Minimize(cp.sum(cp.multiply(c,x)))
con= [cp.sum(x, axis=0)==d,
      cp.sum(x, axis=1)<=e]
prob = cp.Problem(obj, con)
prob.solve()
print("最优值为:",prob.value)
print("最优解为：\n",x.value)
xd=pd.DataFrame(x.value)
xd.to_excel("data5_5_4.xlsx")  #数据写到Excel文件，便于做表使用

# # x=cp.Variable((6,8), pos=True)
# x1 = cp.Variable((3,8), pos=True)
# x2 = cp.Variable((3,8), integer=True)
# x = 10*cp.vstack((x1,x2))
# obj=cp.Minimize(cp.sum(cp.multiply(c,x)))
# con= [cp.sum(x, axis=0)==d,
#       cp.sum(x, axis=1)<=e, x2>=0]
# prob = cp.Problem(obj, con)
# prob.solve(solver='GUROBI')
# print("最优值为:",prob.value)
# print("最优解为：\n",x.value)
# xd=pd.DataFrame(x.value)
# xd.to_excel("data5_5_4_2.xlsx")  #数据写到Excel文件，便于做表使用
