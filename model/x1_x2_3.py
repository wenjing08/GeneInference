import torch
import torch.nn as nn
import sys
sys.path.append('/home/lvfengmao/wanggang/GeneInference/dense/model')
from utils.MyActivation import MyActivation, power_x


class x1_x2_3(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(x1_x2_3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            MyActivation(power_x(1, 2)))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            MyActivation(power_x(1, 2)))
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            MyActivation(power_x(1, 2)))
        self.layer4 = nn.Sequential(
            nn.Linear(in_size + hidden_size + hidden_size + hidden_size, out_size))
    def forward(self, x):
        layer1_out = self.layer1(x)

        tmp1 = torch.cat((x, layer1_out), 1)
        layer2_out = self.layer2(layer1_out)

        tmp2 = torch.cat((tmp1, layer2_out), 1)
        layer3_out = self.layer3(layer2_out)

        tmp3 = torch.cat((tmp2, layer3_out), 1)

        output = self.layer4(tmp3)
        return output