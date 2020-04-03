import torch
import torch.nn as nn



class p_relu_2(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(p_relu_2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(hidden_size, out_size))

    def forward(self, x):

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        output = self.output(layer2_out)

        return output