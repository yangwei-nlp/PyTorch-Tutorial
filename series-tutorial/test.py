import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],  # h 0
              [0, 1, 0, 0, 0],  # i 1
              [1, 0, 0, 0, 0],  # h 0
              [0, 0, 1, 0, 0],  # e 2
              [0, 0, 0, 1, 0],  # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [1, 0, 2, 3, 3, 4]

inputs = Variable(torch.Tensor(x_one_hot))
labels = Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1  # one sentence
sequence_length = 6  # |ihello| == 6
num_layers = 1  # one-layer rnn


class RNN(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        self.rnn = nn.RNN(input_size=5, hidden_size=5, batch_first=True)
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        x.view(x.size(0), self.sequence_length, self.input_size)
        
        out, _ = self.rnn(x, h_0)
        return out.view(-1, num_classes)


rnn = RNN(num_classes, input_size, hidden_size, num_layers)
print(rnn)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

for epoch in range(100):
    outputs = rnn(inputs)
    optimizer.zero_grad()
    
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))
    print("Predicted string: ", ''.join(result_str))

print("Learning finished!")
