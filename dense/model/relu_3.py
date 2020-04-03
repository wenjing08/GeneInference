import torch
import torch.nn as nn



class relu_3(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(relu_3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(in_size + hidden_size + hidden_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(in_size + hidden_size + hidden_size + hidden_size, out_size))

    def forward(self, x):

        layer1_out = self.layer1(x)

        tmp1 = torch.cat((x, layer1_out), 1)
        layer2_out = self.layer2(tmp1)

        tmp2 = torch.cat((tmp1, layer2_out), 1)
        layer3_out = self.layer3(tmp2)

        tmp3 = torch.cat((tmp2, layer3_out), 1)
        output = self.output(tmp3)

        return output