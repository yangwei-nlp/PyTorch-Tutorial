import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

# 1. 每个句子5种类别，适用于文本分类
criterion = nn.CrossEntropyLoss()
pred = torch.randn(3, 5, requires_grad=True)
target = torch.LongTensor([3, 2, 0])  # shape:[3,]
output = criterion(pred, target)
output.backward()

# 2. 对序列标注等任务的loss计算
criterion = nn.CrossEntropyLoss(reduction='none')
# reduction='none'很关键，对每个分类计算交叉熵后并不直接平均，需要和mask相乘

pred = torch.randn(32, 256, 13)
tar = torch.LongTensor(
    np.random.randint(low=0, high=13, size=(32, 256), dtype=int))
mask = torch.LongTensor(
    np.random.randint(low=0, high=2, size=(32, 256),
                      dtype=int))  # 这里的mask只是举例哈

loss = criterion(pred.view(-1, 13),
                 tar.view(-1)).view_as(mask)  # 很关键：将所有词拉平然后降低为2维，在2维数据上
final_loss = torch.sum(loss.mul(mask.float())) / torch.sum(
    mask.float())  # 所有未mask token的平均交叉熵损失

# 3. sigmoid二分类

# 3.1
# nn.BCEWithLogitsLoss 直接将nn.Linear的输出作为输入
# nn.BCEWithLogitsLoss 内部会对nn.Linear的输出做sigmoid
loss = nn.BCEWithLogitsLoss()

inputs = torch.randn(3)
preds = nn.Linear(3, 3)(inputs)
y_true = torch.empty(3).random_(2)

loss(preds, y_true)


# 3.1
# nn.BCELoss 的输入是sigmoid的输出
loss = nn.BCELoss()

inputs = torch.randn(3)
preds = nn.Linear(3, 3)(inputs)
preds = nn.Sigmoid()(preds)
y_true = torch.empty(3).random_(2)

loss(preds, y_true)


