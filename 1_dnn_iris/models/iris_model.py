import torch
import torch.nn as nn
import numpy as np

class MyNetwork(nn.Module):
    # 一个适用于iris的神经网络
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 5)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(5, 3)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)

        x = self.softmax(x)

        return x


    

