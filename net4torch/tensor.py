"""
最近碰到大量的关于tensor的概念和操作，所以非常有必要来重点研究一下tensor相关操作和概念
"""
import torch
import torch.nn as nn

# good articles
# https://zhpmatrix.github.io/2019/03/09/confusing-operation-pytorch/

# reshape的坑
tensor = torch.randn((2, 5))
print(tensor)

#
new_tensor = torch.reshape(tensor, (5, 2))
print(new_tensor)

# 可见我们的reshape是按照行来进行reshape(变形)的
# 另外尽量用torch.reshape而非data.reshape


# torch.masked_select
a = torch.Tensor([[4, 5, 7], [3, 9, 8], [2, 3, 4]])
b = torch.Tensor([[1, 1, 0], [0, 0, 1], [1, 0, 1]]).type(torch.ByteTensor)
c = torch.masked_select(a, b)
print(c)

# torch.gather
t = torch.rand(3, 4, 5)
print(t)
torch.gather(t, 1, torch.LongTensor([[[1, 1, 1, 1, 1]], [[2, 2, 2, 2, 2]], [[3, 3, 3, 3, 3]]]))
print(233)

# tensor的索引
# https://blog.csdn.net/SHU15121856/article/details/87810372

a = torch.randn(3, 4)
print(a)
x = a.ge(0.5)
print(x)
print(a[x].shape)
