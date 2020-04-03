import torch
import torch.nn as nn
import sys
sys.path.append('/home/lvfengmao/wanggang/GeneInference/dense/model')
from utils.MyActivation import MyActivation, power_x, func_exponent_x


class e2_1(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(e2_1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            MyActivation(func_exponent_x))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(dropout_rate),
            MyActivation(func_exponent_x))
        self.layer3 = nn.Sequential(
            nn.Linear(in_size + hidden_size + 1000, out_size))
    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        output = self.layer3(torch.cat((torch.cat((x, layer1_out),1), layer2_out), 1))
        return output