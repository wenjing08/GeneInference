import torch
import torch.nn as nn



class relu_1(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(relu_1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(in_size + hidden_size, out_size))

    def forward(self, x):

        layer1_out = self.layer1(x)

        tmp1 = torch.cat((x, layer1_out), 1)
        output = self.output(tmp1)

        return output