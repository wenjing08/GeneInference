import torch
import torch.nn as nn
import sys
sys.path.append('/home/lvfengmao/wanggang/GeneInference/dense/model')
from utils.MyActivation import MyActivation, power_x, func_exponent_x

class x1_x2_2_e(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(x1_x2_2_e, self).__init__()

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
            MyActivation(func_exponent_x))
        self.layer4 = nn.Sequential(
            nn.Linear(in_size + hidden_size + hidden_size, out_size))
    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        output = self.layer4(torch.cat((torch.cat((x, layer1_out),1), layer3_out), 1))
        return output