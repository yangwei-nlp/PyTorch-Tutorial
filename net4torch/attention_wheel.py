# -*- coding: utf-8 -*-
"""
Description :   实现多头【自注意力Attention机制】，注意，这是个轮子，不建议直接使用，建议学习
     Author :   Yang
       Date :   2020/3/21
"""
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
      q: 请求的形状 == (..., seq_len_q, depth)
      k: 主键的形状 == (..., seq_len_k, depth)
      v: 数值的形状 == (..., seq_len_v, depth_v)
      mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
      输出，注意力权重
    """
    # 如果是多维矩阵，torch,matmul只会在后两个维度相乘
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # 每个词和其他词的点乘
    d_k = k.shape[-1]
    scaled_attention_logits = matmul_qk / (d_k ** 0.5)

    if mask is not None:
        # 由于是矩阵相乘，所以只在后两个维度运算。若mask某个维度数据不够，则
        # 会对mask中为1维的维度广播计算
        scaled_attention_logits += (mask * -1e9)

    attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)
    output = torch.matmul(attention_weights, v)  # 对值向量v加权求和

    return output, attention_weights


def create_padding_mask(seq):
    "0代表padding符，也即需要mask的字符"
    mask = seq == 0
    # 由于padding mask机制只需要掩盖为0的位置，所以后续会使用广播机制相加
    return torch.unsqueeze(mask, 1)


def create_look_ahead_mask(seq):
    # output shape: (seq_len_q, seq_len_k)
    length = seq.shape[1]
    return 1 - torch.triu(torch.ones(length, length)).transpose(1, 0)


"""

inp = torch.randn(32, 512)

padding_mask = create_padding_mask(inp)  # padding符 mask
look_ahead_mask = create_look_ahead_mask(inp)  # 前瞻 mask

multi_embed_x = torch.rand(32, 8, 512, 96)

embed_x = torch.rand(32, 512, 768)
output, attention_weights = scaled_dot_product_attention(
    multi_embed_x, multi_embed_x, multi_embed_x, look_ahead_mask)

"""


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # 每个head的维度

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)

        这里是个坑，不能直接reshape为(batch_size, self.num_heads, -1, self.depth)
        因为reshape会改变数据的相对位置，而转置只是改变了数据的摆放位置
        """
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2)
        # concat注意力
        concat_attention = torch.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


# """
mha = MultiHeadAttention(768, 8)

inp = torch.randn(32, 512, 768)
look_ahead_mask = create_look_ahead_mask(inp)  # 前瞻 mask

attention, weights = mha(inp, inp, inp, look_ahead_mask)
# """
