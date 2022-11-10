
\ comments start with \
\ this is an example.
Maximize
70 x0 + 50 x1 + 60 x2
\ 70 x0 + 50.0001 x1 + 60 x2
Subject To
a0: 2 x0 + 4 x1 + 3 x2 <= 150
a1: 3 x0 + 1 x1 + 5 x2 <= 160
    7 x0 + 3 x1 + 5 x2 <= 200
Bounds
x0 >= 0
x1 >= 0
x2 >= 0
End


