import torch
import torch.nn as nn



class p_relu_1(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(p_relu_1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(hidden_size, out_size))

    def forward(self, x):

        layer1_out = self.layer1(x)

        output = self.output(layer1_out)

        return output