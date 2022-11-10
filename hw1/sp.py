
import sympy as sp
sp.var('k',positive=True, integer=True)
L = sp.Matrix([
    [sp.Rational(6, 10), sp.Rational(1, 10), sp.Rational(3, 10)],
    [sp.Rational(1, 10), sp.Rational(9, 10), 0],
    [sp.Rational(3, 10), 0, sp.Rational(7, 10)]
])


val = L.eigenvals()             #求特征值
# print(val)                    # 输出特征值
vec = L.eigenvects()            #求特征向量
# print(vec)                    # 输出特征向量
P, D = L.diagonalize()          #把L相似对角化
# print(P)
# print(D)
Lk = P @ (D ** k) @ (P.inv())
F = Lk @ sp.Matrix([2, 1, 1])
x = []
x.append(sp.simplify(F[0]))
x.append(sp.simplify(F[1]))
x.append(sp.simplify(F[2]))
print(x[0])                     # 输出解析解
print(x[1])
print(x[2])