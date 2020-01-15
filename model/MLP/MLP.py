import torch
import torch.nn as nn
import torch.nn.functional as F


class Config():
    def __init__(self):
        self.input_size = None
        self.train_path = None
        self.hidden_size = None



class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_size, config.hidden_size_1)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size_1, config.hidden_size_2)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(config.hidden_size_2, config.num_classes)

    def forward(self, x):
        # Fully connected neural network with two hidden layers
        out = self.fc1(x)
        out = self.relu_1(out)
        out = self.fc2(out)
        out = self.relu_2(out)
        out = self.fc3(out)
        return out
