import torch
import torch.nn as nn
import numpy as np

a = torch.rand(2, 4)
b = torch.rand(2, 5, 4)
c = b.transpose(-2, -1)  # 2,4,5
out = torch.matmul(a, c)  # 2,2,5
print(23333)
"""
32*768---32*256*768
32*1*768---32*768*256

32*256

"""

torch.manual_seed(22)
a = torch.rand(2, 1, 4)
b = torch.rand(2, 4, 3)
out = torch.matmul(a, b)
print(out)




one_a = [0.3659, 0.7025, 0.3104, 0.0097]
one_b = [0.8174, 0.6874, 0.9066, 0.7978]

two_a = [0.6577, 0.1947, 0.9506, 0.6887]
two_b = [0.6949, 0.4951, 0.7457, 0.1708]
ret = 0
for i in range(4):
    ret += one_a[i] * one_b[i]
print(ret, out[0][0][0])

ret_two = 0
for i in range(4):
    ret_two += two_a[i] * two_b[i]
print(ret_two, out[1][0][0])

"""
上面这个例子中，2,1,4 shape 和2,4,3 shape进行点乘，
首先，前者中第一个1,4向量和后者第一个4,3向量点乘
其次，前者中第二个1,4向量和后者第二个4,3向量点乘
最终得到2,1,3 shape
"""
