\ comments start with \
\ this is an example.
\ x24, x33, x34, x42, x43, x44 = 0
Min
2800 x11 + 2800 x21 + 2800 x31 + 2800 x41  +  4500 x12 + 4500 x22 + 4500 x32  + 6000 x13 + 6000 x23  + 7300 x14
Subject To
  x11 + x12 + x13 + x14 >= 15 
  x12 + x13 + x14 + x21 + x22 + x23 >= 10
  x13 + x14 + x22 + x23 + x31 + x32 >= 20
  x14 + x23 + x32 + x41 >= 12
Bounds
x11 >= 0
x12 >= 0
x13 >= 0
x14 >= 0
x21 >= 0
x22 >= 0
x23 >= 0
x31 >= 0
x32 >= 0
x41 >= 0
End


