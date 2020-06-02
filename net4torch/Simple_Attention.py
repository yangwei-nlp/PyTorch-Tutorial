"""
这里的attention不是多头注意力
"""

import torch
import torch.nn as nn
import numpy as np


def dot_product_attention(subject_vec, sentence_vec, mask=None):
    """计算主语位置的语义向量和其他位置的向量的attention后的加权向量，
       试图让模型在预测时更多的关注这个已经预测出来的主语。
    """
    # 如果是多维矩阵，torch,matmul只会在后两个维度相乘
    matmul_qk = torch.matmul(subject_vec, sentence_vec.transpose(-2, -1))  # 每个词和其他词的点乘
    d_k = sentence_vec.shape[-1]
    scaled_attention_logits = matmul_qk / (d_k ** 0.5)

    if mask is not None:
        # 由于是矩阵相乘，所以只在后两个维度运算。若mask某个维度数据不够，则
        # 会对mask中为1维的维度广播计算
        add_vals = (mask * -1e9).unsqueeze(1)
        scaled_attention_logits += add_vals

    attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)
    output = torch.matmul(attention_weights, sentence_vec)  # 对值向量v加权求和

    return output


subject_vec = torch.randn(32, 1, 768)
sentence_vec = torch.rand(32, 256, 768)


tmp_tokens = np.random.randint(low=0, high=200, size=(32, 256))
tmp_tokens[:, -1] = 0
token_ids = torch.tensor(tmp_tokens)
# v = sentence_vec

mask = (token_ids == 0).float()

out = dot_product_attention(subject_vec, sentence_vec, mask)

print(123)
