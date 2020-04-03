import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/home/wanggang/projects/GeneInference/dense/')
from model.relu_3 import relu_3
from model.relu_2 import relu_2
from model.relu_1 import relu_1

model = str(sys.argv[1])
input_size = 943
hidden_size = 6000
output_size = 4760
dataset = '9520-2'

# load x & y
X_tr = torch.from_numpy(np.array(np.load('../original_dataset/bgedv2_X_tr_float64.npy'))).type(torch.FloatTensor).cuda()
Y_tr = np.array(np.load('../original_dataset/bgedv2_Y_tr_4760-9520_float64.npy'))
X_va = torch.from_numpy(np.array(np.load('../original_dataset/bgedv2_X_va_float64.npy'))).type(torch.FloatTensor).cuda()
Y_va = np.array(np.load('../original_dataset/bgedv2_Y_va_4760-9520_float64.npy'))

print(X_tr.shape)
print(X_va.shape)

pre_tr = np.zeros((7, 88807, 4760))
pre_va = np.zeros((7, 11101, 4760))
# 定义五个网络
net_0 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_0.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-"+str(dataset)+"-0.pt"))
# net_0 = nn.DataParallel(net_0, device_ids=[0])
# net_0.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-0.1-0_GEO_3000*2.pt"))


pre_tr[0] = np.array(net_0(X_tr).detach().cpu().numpy())
pre_va[0] = np.array(net_0(X_va).detach().cpu().numpy())
torch.cuda.empty_cache()
print("0 over")


net_1 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_1.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-"+str(dataset)+"-1.pt"))
# net_1 = nn.DataParallel(net_1, device_ids=[1])
# net_1.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-0.1-1_GEO_3000*2.pt"))
pre_tr[1] = np.array(net_1(X_tr).detach().cpu().numpy())
pre_va[1] = np.array(net_1(X_va).detach().cpu().numpy())
torch.cuda.empty_cache()
print("1 over")


net_2 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_2.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size) +"-"+str(dataset)+"-2.pt"))
# net_2 = nn.DataParallel(net_2, device_ids=[2])
# net_2.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-0.1-2_GEO_3000*2.pt"))
pre_tr[2] = np.array(net_2(X_tr).detach().cpu().numpy())
pre_va[2] = np.array(net_2(X_va).detach().cpu().numpy())
torch.cuda.empty_cache()
print("2 over")


net_3 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_3.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-3.pt"))
# net_3 = nn.DataParallel(net_3, device_ids=[0])
# net_3.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-0.1-3_GEO_3000*2.pt"))
pre_tr[3] = np.array(net_3(X_tr).detach().cpu().numpy())
pre_va[3] = np.array(net_3(X_va).detach().cpu().numpy())
torch.cuda.empty_cache()
print("3 over")


net_4 = globals()[model](input_size, hidden_size, output_size, 0.1).cuda()
net_4.load_state_dict(torch.load("../res/external_res/from_swufe/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-4.pt"))
# net_4 = nn.DataParallel(net_4, device_ids=[1])
# net_4.load_state_dict(torch.load("../res/dense/"+str(model)+"-"+str(hidden_size)+"-" +str(dataset)+"-0.1-4_GEO_3000*2.pt"))
pre_tr[4] = np.array(net_4(X_tr).detach().cpu().numpy())
pre_va[4] = np.array(net_4(X_va).detach().cpu().numpy())
torch.cuda.empty_cache()
print("4 over")

pre_tr[5] = (pre_tr[0] + pre_tr[1] + pre_tr[2] + pre_tr[3] + pre_tr[4])/5
pre_va[5] = (pre_va[0] + pre_va[1] + pre_va[2] + pre_va[3] + pre_va[4])/5

tmp_tr = np.zeros((5,88807,4760))
tmp_tr[0] = pre_tr[0]
tmp_tr[1] = pre_tr[1]
tmp_tr[2] = pre_tr[2]
tmp_tr[3] = pre_tr[3]
tmp_tr[4] = pre_tr[4]
pre_tr[6] = np.median(tmp_tr, axis=0)

tmp_va = np.zeros((5,11101,4760))
tmp_va[0] = pre_va[0]
tmp_va[1] = pre_va[1]
tmp_va[2] = pre_va[2]
tmp_va[3] = pre_va[3]
tmp_va[4] = pre_va[4]
pre_va[6] = np.median(tmp_va, axis=0)

MSE_tr = np.zeros((7))
MSE_va = np.zeros((7))
# MAE_tr = np.zeros((7))
MAE_va = np.zeros((7))

print("lamda_k^2:\n")
for i in range(0, 7):
    MSE_tr[i] = np.square(pre_tr[i] - Y_tr).mean()
    print(MSE_tr[i])
print("D_k:\n")
for i in range(0, 7):
    MSE_va[i] = np.square(pre_va[i] - Y_va).mean()
    print(MSE_va[i])
#
# for i in range(0, 7):
#     MAE_tr[i] = np.abs(pre_tr[i] - Y_tr).mean()

print("MAE_k:\n")
for i in range(0, 7):
    MAE_va[i] = np.abs(pre_va[i] - Y_va).mean()
    print(MAE_va[i])
# L1
delta_l1 = np.zeros((7))
w_l1 = np.zeros((7))
delta_l1_all = np.zeros((1))
for i in range(0, 7):
    delta_l1[i] = np.exp(-11101/2 * (np.log(MSE_tr[i]/MSE_tr[5])
                                     + 2 * (MAE_va[i]/np.sqrt(MSE_tr[i])
                                            - MAE_va[5]/np.sqrt(MSE_tr[5]))))
    delta_l1_all += delta_l1[i]
print("w_l1_k:\n")
for i in range(0,7):
    w_l1[i] = delta_l1[i]/delta_l1_all
    print(str(w_l1[i]))

# L2
delta_l2 = np.zeros((7))
w_l2 = np.zeros((7))
delta_l2_all = np.zeros((1))
for i in range(0,7):
    delta_l2[i] = np.exp(-11101/2 * (np.log(MSE_tr[i]/MSE_tr[5])
                                     + (MSE_va[i]/MSE_tr[i] - MSE_va[5]/MSE_tr[5])))
    delta_l2_all += delta_l2[i]
print("w_l2_k:\n")
for i in range(0,7):
    w_l2[i] = delta_l2[i]/delta_l2_all
    print(str(w_l2[i]))

