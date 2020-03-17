#!/usr/bin/env python
# coding: utf-8

# ## 数据预处理

# ### 样本选取

# 主要内容：将原始的数据进行选择：仅仅选择5个类别的数据，每个类别仅仅选择5000条数据

# In[12]:


'''
从原数据中选取部分数据；
选取数据的title前两个字符在字典WantedClass中；
且各个类别的数量为WantedNum
'''
import jieba
import json

TrainJsonFile = 'baike_qa2019/baike_qa_train.json'
MyTainJsonFile = 'baike_qa2019/my_traindata.json'
StopWordFile = 'stopword.txt'

WantedClass = {'教育': 0, '健康': 0, '生活': 0, '娱乐': 0, '游戏': 0}
WantedNum = 5000
numWantedAll = WantedNum * 5


def main():
    # read all lines in one time
    Datas = open(TrainJsonFile, 'r', encoding='utf_8').readlines()
    f = open(MyTainJsonFile, 'w', encoding='utf_8')

    numInWanted = 0
    for line in Datas:  # Datas is a list, each element represents the one line of TrainJsonFile and its type is str
        data = json.loads(line)
        # convert string to dict in order to take out category data in the next line code
        cla = data['category'][0:2]  # take out the class
        if cla in WantedClass and WantedClass[cla] < WantedNum:
            json_data = json.dumps(data, ensure_ascii=False)  # no order when writing
            # convert dict to string in order to "write" in the next line code
            f.write(json_data)
            f.write('\n')
            WantedClass[cla] += 1
            numInWanted += 1
            if numInWanted >= numWantedAll:
                break


# In[13]:


main()


# In[ ]:





# ### 生成词表

# In[14]:


'''
将训练数据使用jieba分词工具进行分词。并且剔除stopList中的词。
得到词表：
        词表的每一行的内容为：词 词的序号 词的频次
'''


import json
import jieba
from tqdm import tqdm

trainFile = 'baike_qa2019/my_traindata.json'
stopwordFile = 'stopword.txt'
wordLabelFile = 'wordLabel.txt'
lengthFile = 'length.txt'


def read_stopword(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')
    # data is a list, each element is a char or string which represents a stop word

    return data


def main():
    worddict = {}
    stoplist = read_stopword(stopwordFile)
    datas = open(trainFile, 'r', encoding='utf_8').read().split('\n')
    # filter means delete None values, such as " "
    datas = list(filter(None, datas))
    data_num = len(datas)
    len_dic = {}
    for line in datas:
        line = json.loads(line)
        title = line['title']
        title_seg = jieba.cut(title, cut_all=False)  # 默认模式
        length = 0  # length means in one title there are how many words that are not stopword
        for w in title_seg:
            if w in stoplist:
                continue
            length += 1
            if w in worddict:  # worddict means all not stop word, its value means frequency
                worddict[w] += 1
            else:
                worddict[w] = 1
        if length in len_dic:
            # each element in len_dict means there are how many same length titles
            len_dic[length] += 1
        else:
            len_dic[length] = 1

    wordlist = sorted(worddict.items(), key=lambda item: item[1], reverse=True)
    f = open(wordLabelFile, 'w', encoding='utf_8')
    ind = 0
    for t in wordlist:
        d = t[0] + ' ' + str(ind) + ' ' + str(t[1]) + '\n'
        ind += 1
        f.write(d)

    for k, v in len_dic.items():
        # convert frequency to percentage
        len_dic[k] = round(v * 1.0 / data_num, 3)
    len_list = sorted(len_dic.items(), key=lambda item: item[0], reverse=True)
    f = open(lengthFile, 'w')
    for t in len_list:
        d = str(t[0]) + ' ' + str(t[1]) + '\n'
        f.write(d)


# In[15]:


main()


# In[ ]:





# ### 将中文标题转化为数字向量

# In[ ]:





# In[19]:


data = open(labelFile, 'r', encoding='utf_8').read().split('\n')


# In[20]:


data


# In[21]:


label_w2n = {}
label_n2w = {}
for line in data:
    line = line.split(' ')
    name_w = line[0]
    name_n = int(line[1])
    label_w2n[name_w] = name_n
    label_n2w[name_n] = name_w


# In[22]:


label_w2n


# In[23]:


label_n2w


# In[18]:


trainFile = 'baike_qa2019/my_traindata.json'
stopwordFile = 'stopword.txt'
wordLabelFile = 'wordLabel.txt'
trainDataVecFile = 'traindata_vec.txt'
maxLen = 20

labelFile = 'label.txt'


# In[26]:


datas = open(wordLabelFile, 'r', encoding='utf_8').read().split('\n')
print(len(datas))


# In[27]:


datas = list(filter(None, datas))


# In[28]:


len(datas)


# In[24]:


def get_worddict(file):
    datas = open(file, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    word2ind = {}
    for line in datas:
        line = line.split(' ')
        word2ind[line[0]] = int(line[1])  # word to int

    ind2word = {word2ind[w]: w for w in word2ind}  # int to word
    return word2ind, ind2word


# In[ ]:


get_worddict(wordLabelFile)


# In[30]:


word2ind, ind2word = get_worddict(wordLabelFile)


# In[16]:


import json
import sys
import io
import jieba
import random

# sys.stdout = io.TextIOWrapper(
#     sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码

trainFile = 'baike_qa2019/my_traindata.json'
stopwordFile = 'stopword.txt'
wordLabelFile = 'wordLabel.txt'
trainDataVecFile = 'traindata_vec.txt'
maxLen = 20

labelFile = 'label.txt'


def read_labelFile(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')
    label_w2n = {}
    label_n2w = {}
    for line in data:
        line = line.split(' ')
        name_w = line[0]
        name_n = int(line[1])
        label_w2n[name_w] = name_n
        label_n2w[name_n] = name_w

    return label_w2n, label_n2w


def read_stopword(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')

    return data


def get_worddict(file):
    datas = open(file, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    word2ind = {}
    for line in datas:
        line = line.split(' ')
        word2ind[line[0]] = int(line[1])  # word to int

    ind2word = {word2ind[w]: w for w in word2ind}  # int to word
    return word2ind, ind2word


def json2txt():
    label_dict, label_n2w = read_labelFile(labelFile)
    word2ind, ind2word = get_worddict(wordLabelFile)  # notice: all the words in wordLabelFile represents vocabulary

    traindataTxt = open(trainDataVecFile, 'w')
    stoplist = read_stopword(stopwordFile)
    datas = open(trainFile, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    random.shuffle(datas)
    for line in datas:
        line = json.loads(line)
        title = line['title']
        cla = line['category'][0:2]
        cla_ind = label_dict[cla]  # label int

        title_seg = jieba.cut(title, cut_all=False)  # cut title again
        title_ind = [cla_ind]  # first element in this list is a label int
        for w in title_seg:
            if w in stoplist:
                continue
            title_ind.append(word2ind[w])  # convert title string to title int and append it into a list
        length = len(title_ind)
        # 截长补短
        if length > maxLen + 1:
            title_ind = title_ind[0:21]
        if length < maxLen + 1:
            title_ind.extend([0] * (maxLen - length + 1))
        # every line in datas is a example(with x and y), now turn examples into a new file in a numeric way
        # (splitted with a ",")
        for n in title_ind:
            traindataTxt.write(str(n) + ',')
        traindataTxt.write('\n')


def main():
    json2txt()


# In[17]:


main()


# traindata.txt:
# 其中每一行第一个数字为类别，剩下20个数字为句子内容。这里决定得最大句子长度为20.

# ## 模型搭建

# In[31]:


textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}

import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class textCNN(nn.Module):
    def __init__(self, param):
        super(textCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = param['kernel_num']  # output chanel size
        kernel_size = param['kernel_size']
        vocab_size = param['vocab_size']  # length of word dict
        embed_dim = param['embed_dim']  # dimension of embedding vector
        dropout = param['dropout']
        class_num = param['class_num']
        self.param = param
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=1)
        # 注意：在词向量层，词向量参数为随机初始化，然后每次输入一个句子x，模型就可以训练对应参数的向量了
        # here the vocab_size means there are vocab_size unique words
        # here the padding_idx means: word vector of index 1 word in vocab is constant(no study during trainning)
        self.conv11 = nn.Conv2d(in_channels=ci, out_channels=kernel_num, kernel_size=(
            kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))
        # here the code is to initialize the embedding matrix as a given matrix

    @staticmethod
    def conv_and_pool(x, conv):
        # 一个静态方法，跟普通函数没什么区别，与类和实例都没有所谓的绑定关系，它只不过是碰巧存在类中的一个函数而已
        # x: (batch, 1, sentence_length)
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # x.size(2) means take out the 2th value in the shape list
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        x = x.unsqueeze(1)  # this means add a dimension at axis 1, element at this new dimension is surely one
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)
        return logit

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ## 数据加载

# In[35]:


from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np


trainDataFile = 'traindata_vec.txt'
valDataFile = 'valdata_vec.txt'


def get_valdata(file=valDataFile):
    valData = open(valDataFile, 'r').read().split('\n')
    valData = list(filter(None, valData))
    random.shuffle(valData)

    return valData


class textCNN_data(Dataset):
    def __init__(self):
        trainData = open(trainDataFile, 'r').read().split('\n')
        trainData = list(filter(None, trainData))
        random.shuffle(trainData)
        self.trainData = trainData

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        data = self.trainData[idx]
        data = list(filter(None, data.split(',')))
        data = [int(x) for x in data]
        cla = data[0]
        sentence = np.array(data[1:])
        # sentence是自变量x，每个维度的数字代表一个单词，该数字送到nn.embedding（词向量参数随机初始化）层后
        # 根据该数字找到对应的词向量（look-up），然后根据该句话可以训练参数
        # 至此，原始输入sentence的维度发生变化，每维变为一个向量
        return cla, sentence


def textCNN_dataLoader(param):
    dataset = textCNN_data()
    batch_size = param['batch_size']
    shuffle = param['shuffle']
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# In[36]:


dataset = textCNN_data()
cla, sen = dataset.__getitem__(0)

print(cla)
print(sen)


# ## 训练模型

# In[45]:


import torch
import os
import torch.nn as nn
import numpy as np
import time

# from model import textCNN
# import sen2inds
# import textCNN_data

# word2ind, ind2word = sen2inds.get_worddict('wordLabel.txt')
# label_w2n, label_n2w = sen2inds.read_labelFile('label.txt')

word2ind, ind2word = get_worddict('wordLabel.txt')
label_w2n, label_n2w = read_labelFile('label.txt')

textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}
dataLoader_param = {
    'batch_size': 128,
    'shuffle': True,
}


def main():
    # init net
    print('init net...')
    net = textCNN(textCNN_param)
    weightFile = 'weight.pkl'  # file storages weight
    if os.path.exists(weightFile):
        print('load weight')
        net.load_state_dict(torch.load(weightFile))
    else:
        net.init_weight()
    print(net)

#     net.cuda()

    # init dataset
    print('init dataset...')
    dataLoader = textCNN_dataLoader(dataLoader_param)
#     valdata = get_valdata()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    log = open('log_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log.write('epoch step loss\n')
    log_test = open('log_test_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')
    log_test.write('epoch step test_acc\n')
    print("training...")
    for epoch in range(100):
        for i, (clas, sentences) in enumerate(dataLoader):
            optimizer.zero_grad()
            sentences = sentences.type(torch.LongTensor)#.cuda()
            clas = clas.type(torch.LongTensor)#.cuda()
            out = net(sentences)
            loss = criterion(out, clas)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print("epoch:", epoch + 1, "step:",
                      i + 1, "loss:", loss.item())
                data = str(epoch + 1) + ' ' + str(i + 1) +                     ' ' + str(loss.item()) + '\n'
                log.write(data)
        print("save model...")
        torch.save(net.state_dict(), weightFile)
        torch.save(net.state_dict(), "model\{}_model_iter_{}_{}_loss_{:.2f}.pkl".format(
            time.strftime('%y%m%d%H'), epoch, i, loss.item()))  # current is model.pkl
        print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())


# In[47]:


main()


# ## 模型测试

# In[ ]:


import torch
import os
import torch.nn as nn
import numpy as np
import time

from model import textCNN
import sen2inds

word2ind, ind2word = sen2inds.get_worddict('wordLabel.txt')
label_w2n, label_n2w = sen2inds.read_labelFile('label.txt')

textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}


def get_valData(file):
    datas = open(file, 'r').read().split('\n')
    datas = list(filter(None, datas))

    return datas


def parse_net_result(out):
    score = max(out)
    label = np.where(out == score)[0][0]

    return label, score


def main():
    # init net
    print('init net...')
    net = textCNN(textCNN_param)
    weightFile = 'textCNN.pkl'
    if os.path.exists(weightFile):
        print('load weight')
        net.load_state_dict(torch.load(weightFile))
    else:
        print('No weight file!')
        exit()
    print(net)

    net.cuda()
    net.eval()

    numAll = 0
    numRight = 0
    testData = get_valData('valdata_vec.txt')
    for data in testData:
        numAll += 1
        data = data.split(',')
        label = int(data[0])
        sentence = np.array([int(x) for x in data[1:21]])
        sentence = torch.from_numpy(sentence)
        predict = net(sentence.unsqueeze(0).type(
            torch.LongTensor).cuda()).cpu().detach().numpy()[0]
        label_pre, score = parse_net_result(predict)
        if label_pre == label and score > -100:
            numRight += 1
        if numAll % 100 == 0:
            print('acc:{}({}/{})'.format(numRight / numAll, numRight, numAll))


if __name__ == "__main__":
    main()


# In[3]:


import torch
from torch import nn
from torch.autograd import Variable
# 定义词嵌入
embeds = nn.Embedding(100, 5) # 2 个单词，维度 5
# 得到词嵌入矩阵,开始是随机初始化的
torch.manual_seed(1)
# embeds.weight
# 输出结果：
# Parameter containing:
# -0.8923 -0.0583 -0.1955 -0.9656  0.4224
#  0.2673 -0.4212 -0.5107 -1.5727 -0.1232
# [torch.FloatTensor of size 2x5]


# In[4]:


input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))


# In[6]:


embeds(input)


# In[ ]:




