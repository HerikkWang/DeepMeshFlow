import torch
import torch.nn as nn
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxp = nn.MaxPool2d(kernel_size=2, ceil_mode=False)

    def forward(self, x):
        x = self.maxp(x)
        return x

square_size = 5
inputs = torch.randn(1, 1, square_size, square_size)
for i in range(square_size):
    inputs[0][0][i] = i * torch.ones(square_size)
inputs = Variable(inputs)
print(inputs)

net = Net()
outputs = net(inputs)
print(outputs.size())
print(outputs)