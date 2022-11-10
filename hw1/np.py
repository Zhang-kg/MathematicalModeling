# k=1000时求数值解
# import numpy as np
# L = np.array([[0.6, 0.1, 0.3], [0.1, 0.9, 0], [0.3, 0, 0.7]])
# x = np.array([2, 1, 1]).T
#
# k = 1000
# Lk = L
# for i in range(k):
#     Lk = Lk.dot(L)
#
# print(Lk.dot(x))


#程序文件ex3_2.py
import numpy as np
import sympy as sp
X0 = np.array([2, 1, 1])
Ls = sp.Matrix([
    [sp.Rational(6, 10), sp.Rational(1, 10), sp.Rational(3, 10)],
    [sp.Rational(1, 10), sp.Rational(9, 10), 0],
    [sp.Rational(3, 10), 0, sp.Rational(7, 10)]
])                                  #符号矩阵
sp.var('lamda')                     #定义符号变量
p = Ls.charpoly(lamda)              #计算特征多项式
w1 = sp.roots(p)                    #计算特征值
w2 = Ls.eigenvals()                 #直接计算特征值
v = Ls.eigenvects()                 #直接计算特征向量
print("特征值为：",w2)
print("特征向量为：\n",v)
P, D = Ls.diagonalize()             #相似对角化
Pinv = P.inv()                      #求逆阵
Pinv = sp.simplify(Pinv)
cc = Pinv @ X0
print('P=\n', P)
print('c=', cc[0])                  # 输出c
print(cc[1])
print(cc[2])