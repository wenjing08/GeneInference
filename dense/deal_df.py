import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/home/wanggang/projects/GeneInference/dense/')
from model.df_relu_3 import df_relu_3
from model.df_relu_2 import df_relu_2
from model.relu_1 import relu_1

model = 'df_relu_2'
input_size = 943
hidden_size = 3000
output_size = 4760
dataset = '0-4760'

# load x & y
X_te = torch.from_numpy(np.array(np.load('../original_dataset/GTEx_X_float64.npy'))).type(torch.FloatTensor).cuda()
Y_te = np.array(np.load('../original_dataset/GTEx_Y_0-4760_float64.npy'))

# 定义五个网络
net_0 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
# net_0 = nn.DataParallel(net_0, device_ids=[0])
net_0.load_state_dict(torch.load("../res/external_res/first_half3000_2/model_params_1.pt"))
pre_0 = net_0(X_te).detach().cpu().numpy()
pre_0 = np.array(pre_0)

net_1 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
# net_1 = nn.DataParallel(net_1, device_ids=[0])
net_1.load_state_dict(torch.load("../res/external_res/first_half3000_2/model_params_2.pt"))
pre_1 = net_1(X_te).detach().cpu().numpy()
pre_1 = np.array(pre_1)

net_2 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
# net_2 = nn.DataParallel(net_2, device_ids=[0])
net_2.load_state_dict(torch.load("../res/external_res/first_half3000_2/model_params_3.pt"))
pre_2 = net_2(X_te).detach().cpu().numpy()
pre_2 = np.array(pre_2)

net_3 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
# net_3 = nn.DataParallel(net_3, device_ids=[0])
net_3.load_state_dict(torch.load("../res/external_res/first_half3000_2/model_params_5.pt"))
pre_3 = net_3(X_te).detach().cpu().numpy()
pre_3 = np.array(pre_3)

# net_4 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
# # net_4 = nn.DataParallel(net_4, device_ids=[0])
# net_4.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-4.pt"))
# pre_4 = net_4(X_te).detach().cpu().numpy()
# pre_4 = np.array(pre_4)
#


pre_0_tmp = pre_0 - Y_te
pre_1_tmp = pre_1 - Y_te
pre_2_tmp = pre_2 - Y_te
pre_3_tmp = pre_3 - Y_te
# pre_4_tmp = pre_4 - Y_te

pre_avg = (pre_0 + pre_1 + pre_2 + pre_3)/4
pre_avg_mean = (pre_avg - Y_te).mean(axis=0)


avg_MAE = np.abs(pre_avg - Y_te).mean()
print(avg_MAE)
print(np.abs(pre_0 - Y_te).mean())
print(np.abs(pre_1 - Y_te).mean())
print(np.abs(pre_2 - Y_te).mean())
print(np.abs(pre_3 - Y_te).mean())



re_mae = (np.abs(pre_0 - pre_avg) +
       np.abs(pre_1 - pre_avg) +
       np.abs(pre_2 - pre_avg) +
       np.abs(pre_3 - pre_avg))/4
std_error2 = re_mae.mean()
print(std_error2)