import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_hidden = 5):
        super(MLP,self).__init__()
        self.num_hidden = num_hidden
        self.layer1 = nn.Linear(2,self.num_hidden)
        self.layer2 = nn.Linear(self.num_hidden,self.num_hidden)
        self.layer3 = nn.Linear(self.num_hidden,1)

    def forward(self,input):
        activation1 = F.relu(self.layer1(input))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = torch.sigmoid(torch.flatten(self.layer3(activation2)))
        return activation3

    