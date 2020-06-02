""" 计算precision、recall和f1。如果任务是多分类，计算这些指标这很重要。 """
"""
例子：
y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
# precision
cat: 0.66
dog: 0
pig: 0
macro: 0.22
# recall
cat: 1
dog: 0
pig: 0
macro：0.33
# f1
cat: 2\*0.66\*1/(1+0.66) = 0.8
dog: 0
pig: 0
macro: 0.26
"""
from sklearn.metrics import precision_recall_fscore_support

y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])

# 注意average选择
precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(23333)
