import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/home/wanggang/projects/GeneInference/dense/')
from model.relu_3 import relu_3
from model.relu_2 import relu_2
from model.relu_1 import relu_1

model = str(sys.argv[1]) # relu_3
input_size = 943
hidden_size = int(sys.argv[2]) # 3000
output_size = 4760
dataset = str(sys.argv[3]) # 0-4760
noise = str(sys.argv[4]) # noise or no_noise

# load x & y
X_te = torch.from_numpy(np.array(np.load('../original_dataset/bgedv2_X_te_float64.npy'))).type(torch.FloatTensor).cuda()
Y_te = np.array(np.load('../original_dataset/bgedv2_Y_te_'+str(dataset)+'_float64.npy'))

print(X_te.shape)

pre_0_te = np.load("../res/dense/"+str(model)+"-"+str(hidden_size) +"-0.1-"+str(dataset)+"-0_GEO_"+noise+".npy")
pre_1_te = np.load("../res/dense/"+str(model)+"-"+str(hidden_size) +"-0.1-"+str(dataset)+"-1_GEO_"+noise+".npy")
pre_2_te = np.load("../res/dense/"+str(model)+"-"+str(hidden_size) +"-0.1-"+str(dataset)+"-2_GEO_"+noise+".npy")
pre_3_te = np.load("../res/dense/"+str(model)+"-"+str(hidden_size) +"-0.1-"+str(dataset)+"-3_GEO_"+noise+".npy")
pre_4_te = np.load("../res/dense/"+str(model)+"-"+str(hidden_size) +"-0.1-"+str(dataset)+"-4_GEO_"+noise+".npy")

pre_avg = (pre_0_te + pre_1_te + pre_2_te + pre_3_te + pre_4_te)/5
avg_MAE = np.abs(pre_avg - Y_te).mean()
print("avg_MAE:\n")
print(avg_MAE)

MAE0 = np.abs(pre_0_te - Y_te).mean()
MAE1 = np.abs(pre_1_te - Y_te).mean()
MAE2 = np.abs(pre_2_te - Y_te).mean()
MAE3 = np.abs(pre_3_te - Y_te).mean()
MAE4 = np.abs(pre_4_te - Y_te).mean()
print("MAE:\n")
print(MAE0)
print(MAE1)
print(MAE2)
print(MAE3)
print(MAE4)

tmp = np.zeros((5,11101,4760))
tmp[0] = pre_0_te
tmp[1] = pre_1_te
tmp[2] = pre_2_te
tmp[3] = pre_3_te
tmp[4] = pre_4_te
pre_median = np.median(tmp, axis=0)
median_MAE = np.abs(pre_median - Y_te).mean()
print("median_MAE:\n")
print(median_MAE)

tmp2 = (np.abs(pre_0_te - pre_avg) +
       np.abs(pre_1_te - pre_avg) +
       np.abs(pre_2_te - pre_avg) +
       np.abs(pre_3_te - pre_avg) +
       np.abs(pre_4_te - pre_avg))/5
std_error2 = tmp2.mean()
print("re_MAE:\n")
print(std_error2)