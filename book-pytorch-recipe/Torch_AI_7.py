#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)


# In[2]:


word_to_ix = {"data": 0, "science": 1}


# In[3]:


word_to_ix


# In[4]:


embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings


# In[5]:


embeds


# In[6]:


lookup_tensor = torch.tensor([word_to_ix["data"]], dtype=torch.long)
lookup_tensor


# In[7]:


hello_embed = embeds(lookup_tensor)
print(hello_embed)


# In[8]:


CONTEXT_SIZE = 2


# In[9]:


EMBEDDING_DIM = 10


# In[10]:


test_sentence = """The popularity of the term "data science" has exploded in 
business environments and academia, as indicated by a jump in job openings.[32] 
However, many critical academics and journalists see no distinction between data 
science and statistics. Writing in Forbes, Gil Press argues that data science is a 
buzzword without a clear definition and has simply replaced “business analytics” in 
contexts such as graduate degree programs.[7] In the question-and-answer section of 
his keynote address at the Joint Statistical Meetings of American Statistical 
Association, noted applied statistician Nate Silver said, “I think data-scientist 
is a sexed up term for a statistician....Statistics is a branch of science. 
Data scientist is slightly redundant in some way and people shouldn’t berate the 
term statistician.”[9] Similarly, in business sector, multiple researchers and 
analysts state that data scientists alone are far from being sufficient in granting 
companies a real competitive advantage[33] and consider data scientists as only 
one of the four greater job families companies require to leverage big 
data effectively, namely: data analysts, data scientists, big data developers 
and big data engineers.[34]

On the other hand, responses to criticism are as numerous. In a 2014 Wall Street 
Journal article, Irving Wladawsky-Berger compares the data science enthusiasm with 
the dawn of computer science. He argues data science, like any other interdisciplinary 
field, employs methodologies and practices from across the academia and industry, but 
then it will morph them into a new discipline. He brings to attention the sharp criticisms 
computer science, now a well respected academic discipline, had to once face.[35] Likewise, 
NYU Stern's Vasant Dhar, as do many other academic proponents of data science,[35] argues 
more specifically in December 2013 that data science is different from the existing practice 
of data analysis across all disciplines, which focuses only on explaining data sets. 
Data science seeks actionable and consistent pattern for predictive uses.[1] This practical 
engineering goal takes data science beyond traditional analytics. Now the data in those 
disciplines and applied fields that lacked solid theories, like health science and social 
science, could be sought and utilized to generate powerful predictive models.[1]""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


# In[11]:


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)


# In[12]:


losses


# In[13]:


loss_function


# In[14]:


model


# In[15]:


optimizer


# In[16]:


for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!


# In[17]:


CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right


# In[18]:


raw_text = """For the future of data science, Donoho projects an ever-growing 
environment for open science where data sets used for academic publications are 
accessible to all researchers.[36] US National Institute of Health has already announced 
plans to enhance reproducibility and transparency of research data.[39] Other big journals 
are likewise following suit.[40][41] This way, the future of data science not only exceeds 
the boundary of statistical theories in scale and methodology, but data science will 
revolutionize current academia and research paradigms.[36] As Donoho concludes, "the scope 
and impact of data science will continue to expand enormously in coming decades as scientific 
data and data about science itself become ubiquitously available."[36]""".split()


# In[19]:


# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


# In[20]:


class CBOW(nn.Module):

    def __init__(self):
        pass

    def forward(self, inputs):
        pass
# create your model and train.  here are some functions to help you make
# the data ready for use by your module
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

make_context_vector(data[0][0], word_to_ix)  # example


# In[21]:


lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = torch.randn(2, 5)
print(lin(data))  # yes


# In[22]:


data = torch.randn(2, 2)
print(data)
print(F.relu(data))


# In[23]:


# Softmax is also in torch.nn.functional
data = torch.randn(5)
print(data)
print(F.softmax(data, dim=0))
print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
print(F.log_softmax(data, dim=0))  # theres also log_softmax


# In[24]:


lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)


# In[25]:


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("Probability and random variable are integral part of computation ".split(), 
     ["DET", "NN", "V", "DET", "NN"]),
    ("Understanding of the probability and associated concepts are essential".split(), 
     ["NN", "V", "DET", "NN"])
]


# In[26]:


training_data


# In[27]:


word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6


# In[28]:


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[29]:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
model
loss_function
optimizer


# In[30]:


with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)


# In[31]:


for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


# In[32]:


# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)


# In[ ]:




