import time

import numpy as np
import sys
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.modules import Module

'''
1、定义超参数
'''
n_epoch = 200
b_size = 5000

in_size = 943
out_size = 3173

hidden_size = int(sys.argv[1])
print(hidden_size)
dropout_rate = 0.1

l_rate = 5e-4

'''
2、读取数据
'''
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path):

        self.X = np.array(np.load(x_path))
        self.Y = np.array(np.load(y_path))
    def __getitem__(self, index):

        return self.X[index], self.Y[index]
    def getall(self):

        return self.X, self.Y
    def __len__(self):

        return len(self.X)

print('loading data...:')

tr_set = MyDataSet(x_path='../dataset/bgedv2_X_tr_float64.npy', y_path='../dataset/bgedv2_Y_tr_3173_float64.npy')
tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=b_size, shuffle=True)

X_va = torch.from_numpy(np.array(np.load('../dataset/bgedv2_X_va_float64.npy'))).type(torch.FloatTensor).cuda()
Y_va = torch.from_numpy(np.array(np.load('../dataset/bgedv2_Y_va_3173_float64.npy'))).type(torch.FloatTensor).cuda()

X_te = torch.from_numpy(np.array(np.load('../dataset/bgedv2_X_te_float64.npy'))).type(torch.FloatTensor).cuda()
Y_te = torch.from_numpy(np.array(np.load('../dataset/bgedv2_Y_te_3173_float64.npy'))).type(torch.FloatTensor).cuda()

'''
2、定义网络
'''
'''
自定义激活函数
'''
class MyActivation(Module):
    def __init__(self, func=None):
        super(MyActivation, self).__init__()
        self.activation_func = func

    def __call__(self, _input, *args, **kwargs):
        return self.activation_func(_input)


def power_x(k1=1, k2=2):
    """"定义闭包函数
    """

    def func_power_x(_input):

        a = _input[:, 0:500].pow(k1)
        b = _input[:, 500:].pow(k2)
        return torch.cat((a,b), 1)
    return func_power_x


def func_exponent_x(_input):
    return torch.exp(-_input)

'''
网络结构
'''
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

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
            nn.Linear(in_size + hidden_size + hidden_size, out_size))
    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        output = self.layer3(torch.cat((torch.cat((x, layer1_out),1), layer2_out), 1))
        return output

class Mynet1(nn.Module):
    def __init__(self):
        super(Mynet1, self).__init__()

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

net = Mynet1().cuda()
'''
3、定义Loss和优化器
'''
criterion = nn.MSELoss(reduce=True, size_average=False)
optimizer = optim.Adam(net.parameters(), lr=l_rate)
'''
4、开始训练网络
'''
MAE_va_old = 10.0
MAE_te_best = 10.0
MAE_tr_old = 10.0
MAE_te_old = 10.0

outlog = open('../res/dense/net-A1-'+str(hidden_size)+'.log', 'w')
log_str = '\t'.join(map(str, ['epoch', 'MAE_va', 'MAE_te',  'MAE_tr',  'time(sec)']))
print(log_str)
outlog.write(log_str + '\n')
sys.stdout.flush()

for epoch in range(n_epoch):
    for i, data in enumerate(tr_loader, 0):
        t_old = time.time()
        '''
        开始训练了
        '''
        # forward
        net.train()
        x_batch, y_batch = data
        x_batch = x_batch.type(torch.FloatTensor).cuda()
        y_batch = y_batch.type(torch.FloatTensor).cuda()

        tr_outputs = net(x_batch)

        loss = criterion(tr_outputs, y_batch)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        '''
        开始验证了
        '''
        with torch.no_grad():
            net.eval()
            #计算output
            va_outputs = net(X_va)
            te_outputs = net(X_te)

            #计算MAE
            MAE_tr = np.abs(y_batch.detach().cpu().numpy()  - tr_outputs.detach().cpu().numpy() ).mean()
            MAE_va = np.abs(Y_va.detach().cpu().numpy()  - va_outputs.detach().cpu().numpy() ).mean()
            MAE_te = np.abs(Y_te.detach().cpu().numpy()  - te_outputs.detach().cpu().numpy() ).mean()

            MAE_tr_old = MAE_tr
            MAE_va_old = MAE_va
            MAE_te_old = MAE_te

            t_new = time.time()
            l_rate = optim.lr_scheduler
            log_str = '\t'.join(
                map(str, [(epoch * 18) + i + 1, '%.6f' % MAE_va, '%.6f' % MAE_te,
                          '%.6f' % MAE_tr, int(t_new - t_old)]))
            print(log_str)
            outlog.write(log_str + '\n')
            sys.stdout.flush()
            # 保留最优MAE_te,MAE_GTEx
            if MAE_te < MAE_te_best:
                MAE_te_best = MAE_te
    print("epoch %d training over" % epoch)
print('MAE_te_best : %.6f' % (MAE_te_best))
outlog.write('MAE_te_best : %.6f' % (MAE_te_best) + '\n')
outlog.close()
print('Finish Training')
