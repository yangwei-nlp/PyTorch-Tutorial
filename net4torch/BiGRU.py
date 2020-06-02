# -*- coding: utf-8 -*-
"""
Description :   
     Author :   Yang
       Date :   2020/3/22
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bigru = nn.GRU(input_size=768, hidden_size=768, bidirectional=True, batch_first=True)
    
    def forward(self, inputs):
        hidden = self.init_hidden()
        return self.bigru(inputs, hidden)
    
    def init_hidden(self):
        # 注意32是batch数目!!!
        return torch.zeros(2, 32, 768)


model = Model()
inputs = torch.rand(32, 256, 768)
outs, _ = model(inputs)

forward_out, backward_out = torch.split(outs, 768, dim=-1)

print(outs.shape)