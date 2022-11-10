#程序文件ex5_4.py
import cvxpy as cp

x=cp.Variable((4,4),pos=True)
obj=cp.Minimize(2800*sum(x[:,0])+4500*sum(x[:3,1])+
    6000*sum(x[:2,2])+7300*x[0,3])
cons=[sum(x[0,:])>=15,
      sum(x[0,1:])+sum(x[2,:3])>=10,
      sum(x[0,2:])+sum(x[1,1:3])+sum(x[2,:2])>=20,
      x[0,3]+x[1,2]+x[2,1]+[3,0]>=12]
prob=cp.Problem(obj,cons)
prob.solve()
xx= x.value
xx[1,3]=xx[2,2]=xx[2,3]=xx[3,1]=xx[3,2]=xx[3,3]=0
print("最优值为：",prob.value)
print("最优解为：\n",xx)

# x1 = cp.Variable((3,4), pos=True)
# x2 = cp.Variable((1,4), integer=True)
# x = cp.vstack((x1,x2))
# prob=cp.Problem(obj,cons.append(x>=0))
# prob.solve(solver='GUROBI')
# print("最优值为：",prob.value)
# print("最优解为：\n",x.value)