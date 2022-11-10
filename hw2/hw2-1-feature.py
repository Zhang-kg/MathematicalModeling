
import sympy as sp
k = sp.var('k',positive=True, integer=True)
a = sp.Matrix([[0, 1], [1, sp.Rational(1, 2)]])
val = a.eigenvals()                 #求特征值
# print(val)
vec = a.eigenvects()                #求特征向量
print(vec)
P, D = a.diagonalize()              #把a相似对角化
print('P', P)

ak = P @ (D ** k) @ (P.inv())
F = ak @ sp.Matrix([-2, 0])
s = sp.simplify(F[0])
s = sp.latex(s)
print(s)