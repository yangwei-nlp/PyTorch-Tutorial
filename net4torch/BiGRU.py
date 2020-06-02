# -*- coding: utf-8 -*-
"""
Description :   torch中如何使用双向循环神经网络
     Author :   Yang
       Date :   2020/3/22
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 在用BiRNN时，如果我们需要输出size=768，一般做法是将hidden_size设置为768/2
        # 由于输出时会自动将前向状态和后向状态concat，所以最终将会得到768的输出
        self.bigru = nn.GRU(input_size=768,
                            hidden_size=768 // 2,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, inputs):
        hidden = self.init_hidden()
        return self.bigru(inputs, hidden)

    def init_hidden(self):
        # 注意32是batch数目!!!
        return torch.zeros(2, 32, 768 // 2)


model = Model()
inputs = torch.rand(32, 256, 768)
outs, _ = model(inputs)
print(outs.shape)

# forward_out, backward_out = torch.split(outs, 768, dim=-1)
