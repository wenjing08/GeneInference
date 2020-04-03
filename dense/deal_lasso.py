import torch
import torch.nn as nn
import numpy as np
# import sys
#
# xxx = np.random.rand(2,3)
# print(xxx)
# yyy = xxx.copy()
# yyy[yyy>0.5] = 1
# print(yyy)
# yyy[yyy<0.5] = 0
# print(yyy)
#
#
# sys.exit()
dataset = "5"
value = 0
net_0 = nn.Sequential(
    nn.Linear(943, 5)
).cuda()

net_0.load_state_dict(torch.load('../res/dense/Lasso-10.0-'+dataset+'.pt'))

p = list(net_0.parameters())
for i in range(0,5):
    w = p[0][i]
    w[w>value] = 1
    w[w<value] = 0
    print(np.sum(w.detach().cpu().numpy()))