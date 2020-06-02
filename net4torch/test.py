import torch
import torch.nn as nn
import numpy as np

a = torch.rand(3, 4, 5)
starts = torch.LongTensor([[1], [2], [1]])
ends = torch.LongTensor([[3], [3], [2]])

def sum_position_vec(bert_out, starts, ends):
    outputs = torch.zeros(bert_out.size(0), bert_out.size(-1))

    for idx in range(bert_out.size(0)):
        outputs[idx] = torch.sum(bert_out[idx, starts[idx]:ends[idx]+1], dim=0, keepdim=True)

    return outputs.unsqueeze(1)


vecs = sum_position_vec(a, starts, ends)
print(233)