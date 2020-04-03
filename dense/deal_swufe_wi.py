import numpy as np
import torch
import torch.nn as nn
import sys
import random

sys.path.append('/home/wanggang/projects/GeneInference/dense/')
from model.relu_3 import relu_3
from model.relu_2 import relu_2
from model.relu_1 import relu_1


model = str(sys.argv[1])
num = int(sys.argv[2])
input_size = 943
hidden_size = 6000
output_size = 4760
dataset = '9520-2'

# load x & y
# X_va = torch.from_numpy(np.array(np.load('../original_dataset/bgedv2_X_va_float64.npy'))).type(torch.FloatTensor).cuda()
# Y_va = np.array(np.load('../original_dataset/bgedv2_Y_va_4760-9520_float64.npy'))
X_te = torch.from_numpy(np.array(np.load('../original_dataset/bgedv2_X_te_float64.npy'))).type(torch.FloatTensor).cuda()
Y_te = np.array(np.load('../original_dataset/bgedv2_Y_te_4760-9520_float64.npy'))

# print(X_va.shape)
print(X_te.shape)
# 定义五个网络
net_0 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_0.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-"+str(dataset)+"-0.pt"))
# pre_0_va = np.array(net_0(X_va).detach().cpu().numpy())
pre_0_te = np.array(net_0(X_te).detach().cpu().numpy())

net_1 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_1.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-"+str(dataset)+"-1.pt"))
# pre_1_va = np.array(net_1(X_va).detach().cpu().numpy())
pre_1_te = np.array(net_1(X_te).detach().cpu().numpy())

net_2 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_2.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size) +"-"+str(dataset)+"-2.pt"))
# pre_2_va = np.array(net_2(X_va).detach().cpu().numpy())
pre_2_te = np.array(net_2(X_te).detach().cpu().numpy())

net_3 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_3.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-3.pt"))
# pre_3_va = np.array(net_3(X_va).detach().cpu().numpy())
pre_3_te = np.array(net_3(X_te).detach().cpu().numpy())

net_4 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_4.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-4.pt"))
# pre_4_va = np.array(net_4(X_va).detach().cpu().numpy())
pre_4_te = np.array(net_4(X_te).detach().cpu().numpy())

# u0 = (pre_0_va - Y_va).mean(axis=0)
# u1 = (pre_1_va - Y_va).mean(axis=0)
# u2 = (pre_2_va - Y_va).mean(axis=0)
# u3 = (pre_3_va - Y_va).mean(axis=0)
# u4 = (pre_4_va - Y_va).mean(axis=0)
# print("u:\n")
# print(u0)
# print(u1)
# print(u2)
# print(u3)
# print(u4)
#
# s0 = np.square(np.abs(pre_0_te - Y_te -u0)).sum(axis=0).mean()/(11100)
# s1 = np.square(np.abs(pre_1_te - Y_te -u1)).sum(axis=0).mean()/(11100)
# s2 = np.square(np.abs(pre_2_te - Y_te -u2)).sum(axis=0).mean()/(11100)
# s3 = np.square(np.abs(pre_3_te - Y_te -u3)).sum(axis=0).mean()/(11100)
# s4 = np.square(np.abs(pre_4_te - Y_te -u4)).sum(axis=0).mean()/(11100)
#
# se0 = s0/np.sqrt(11101)
# se1 = s1/np.sqrt(11101)
# se2 = s2/np.sqrt(11101)
# se3 = s3/np.sqrt(11101)
# se4 = s4/np.sqrt(11101)
#
# print("se:\n")
# print(se0)
# print(se1)
# print(se2)
# print(se3)
# print(se4)
#
# MAE0 = np.abs(pre_0_te - Y_te).mean()
# MAE1 = np.abs(pre_1_te - Y_te).mean()
# MAE2 = np.abs(pre_2_te - Y_te).mean()
# MAE3 = np.abs(pre_3_te - Y_te).mean()
# MAE4 = np.abs(pre_4_te - Y_te).mean()
#
# print("MAE:\n")
# print(MAE0)
# print(MAE1)
# print(MAE2)
# print(MAE3)
# print(MAE4)
#
# MAE_u0 = np.abs(pre_0_te - Y_te +u0).mean()
# MAE_u1 = np.abs(pre_1_te - Y_te +u1).mean()
# MAE_u2 = np.abs(pre_2_te - Y_te +u2).mean()
# MAE_u3 = np.abs(pre_3_te - Y_te +u3).mean()
# MAE_u4 = np.abs(pre_4_te - Y_te +u4).mean()
#
# print("MAE_u:\n")
# print(MAE_u0)
# print(MAE_u1)
# print(MAE_u2)
# print(MAE_u3)
# print(MAE_u4)

list = random.sample(range(0,11102), num)

mse0 = np.square(pre_0_te[list] - Y_te[list]).mean()
mse1 = np.square(pre_1_te[list] - Y_te[list]).mean()
mse2 = np.square(pre_2_te[list] - Y_te[list]).mean()
mse3 = np.square(pre_3_te[list] - Y_te[list]).mean()
mse4 = np.square(pre_4_te[list] - Y_te[list]).mean()

delta0 = np.exp(num * np.log(mse0 / mse0))
delta1 = np.exp(num * np.log(mse1 / mse0))
delta2 = np.exp(num * np.log(mse2 / mse0))
delta3 = np.exp(num * np.log(mse3 / mse0))
delta4 = np.exp(num * np.log(mse4 / mse0))

S = delta0 + delta1 + delta2 + delta3 + delta4

w0 = delta0 / S
w1 = delta1 / S
w2 = delta2 / S
w3 = delta3 / S
w4 = delta4 / S

print(w0)
print(w1)
print(w2)
print(w3)
print(w4)