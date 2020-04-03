import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/home/wanggang/projects/GeneInference/dense/')
from model.relu_3 import relu_3
from model.relu_2 import relu_2
from model.relu_1 import relu_1

model = 'relu_1'
input_size = 943
hidden_size = 3000
output_size = 4760
dataset = '0-4760'

# load x & y
X_te = torch.from_numpy(np.array(np.load('../second_dataset/GTEx_X_te_float64.npy'))).type(torch.FloatTensor).cuda()
Y_te = np.array(np.load('../second_dataset/GTEx_Y_te_'+str(dataset)+'_float64.npy'))

# 定义五个网络
net_0 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_0 = nn.DataParallel(net_0, device_ids=[0])
net_0.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-0.1-"+str(dataset)+"-0_GEO.pt"))
pre_0 = net_0(X_te).detach().cpu().numpy()
pre_0 = np.array(pre_0)

net_1 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_1 = nn.DataParallel(net_1, device_ids=[0])
net_1.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-0.1-"+str(dataset)+"-1_GEO.pt"))
pre_1 = net_1(X_te).detach().cpu().numpy()
pre_1 = np.array(pre_1)

net_2 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_2 = nn.DataParallel(net_2, device_ids=[0])
net_2.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-0.1-"+str(dataset)+"-2_GEO.pt"))
pre_2 = net_2(X_te).detach().cpu().numpy()
pre_2 = np.array(pre_2)

net_3 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_3 = nn.DataParallel(net_3, device_ids=[0])
net_3.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-0.1-"+str(dataset)+"-3_GEO.pt"))
pre_3 = net_3(X_te).detach().cpu().numpy()
pre_3 = np.array(pre_3)

net_4 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_4 = nn.DataParallel(net_4, device_ids=[0])
net_4.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-0.1-"+str(dataset)+"-4_GEO.pt"))
pre_4 = net_4(X_te).detach().cpu().numpy()
pre_4 = np.array(pre_4)



pre_0_tmp = pre_0 - Y_te
pre_1_tmp = pre_1 - Y_te
pre_2_tmp = pre_2 - Y_te
pre_3_tmp = pre_3 - Y_te
pre_4_tmp = pre_4 - Y_te

pre_avg = (pre_0 + pre_1 + pre_2 + pre_3 + pre_4)/5
pre_avg_mean = (pre_avg - Y_te).mean(axis=0)


avg_MAE = np.abs(pre_avg - Y_te).mean()
print(avg_MAE)
print(np.abs(pre_0 - Y_te).mean())
print(np.abs(pre_1 - Y_te).mean())
print(np.abs(pre_2 - Y_te).mean())
print(np.abs(pre_3 - Y_te).mean())
print(np.abs(pre_4 - Y_te).mean())


mse0 = np.square(pre_0 - Y_te).mean()
mse1 = np.square(pre_1 - Y_te).mean()
mse2 = np.square(pre_2 - Y_te).mean()
mse3 = np.square(pre_3 - Y_te).mean()
mse4 = np.square(pre_4 - Y_te).mean()

delta0 = np.exp(421 * np.log(mse0 / mse0))
delta1 = np.exp(421 * np.log(mse1 / mse0))
delta2 = np.exp(421 * np.log(mse2 / mse0))
delta3 = np.exp(421 * np.log(mse3 / mse0))
delta4 = np.exp(421 * np.log(mse4 / mse0))

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
# tmp = (np.square(pre_0_tmp - pre_avg_mean) +
#        np.square(pre_1_tmp - pre_avg_mean) +
#        np.square(pre_2_tmp - pre_avg_mean) +
#        np.square(pre_3_tmp - pre_avg_mean) +
#        np.square(pre_4_tmp - pre_avg_mean))/5
# std_error = np.sqrt(tmp.mean())
# print(std_error)
#
# tmp1 = (np.square(pre_0 - pre_avg) +
#        np.square(pre_1 - pre_avg) +
#        np.square(pre_2 - pre_avg) +
#        np.square(pre_3 - pre_avg) +
#        np.square(pre_4 - pre_avg))/5
# std_error1 = np.sqrt(tmp1.mean())
# print(std_error1)
#
# tmp2 = (np.abs(pre_0 - pre_avg) +
#        np.abs(pre_1 - pre_avg) +
#        np.abs(pre_2 - pre_avg) +
#        np.abs(pre_3 - pre_avg) +
#        np.abs(pre_4 - pre_avg))/5
# std_error2 = tmp2.mean()
# print(std_error2)