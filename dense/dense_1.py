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


'''
1、定义超参数
'''
n_epoch = 200
b_size = 5000

in_size = 943
out_size = 4760

n_hidden = 3000
dropout_rate = 0.1

file_num = sys.argv[1]
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

tr_set = MyDataSet(x_path='../dataset/bgedv2_X_tr_float64.npy', y_path='../dataset/bgedv2_Y_tr_9520-2_float64.npy')
tr_loader = torch.utils.data.DataLoader(tr_set, batch_size=b_size, shuffle=True)

X_va = torch.from_numpy(np.array(np.load('../dataset/bgedv2_X_va_float64.npy'))).type(torch.FloatTensor).cuda()
Y_va = torch.from_numpy(np.array(np.load('../dataset/bgedv2_Y_va_9520-2_float64.npy'))).type(torch.FloatTensor).cuda()

X_te = torch.from_numpy(np.array(np.load('../dataset/bgedv2_X_te_float64.npy'))).type(torch.FloatTensor).cuda()
Y_te = torch.from_numpy(np.array(np.load('../dataset/bgedv2_Y_te_9520-2_float64.npy'))).type(torch.FloatTensor).cuda()

te_size = X_te.shape[0]
print(te_size)

'''
2、定义网络
'''
class dense_1(nn.Module):
    def __init__(self):
        super(dense_1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, n_hidden),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(in_size + n_hidden, out_size))

    def forward(self, x):

        layer1_out = self.layer1(x)

        tmp1 = torch.cat((x, layer1_out), 1)
        output = self.output(tmp1)

        return output
net = dense_1().cuda()
'''
3、定义Loss和优化器
'''
criterion = nn.MSELoss(reduce=True, size_average=False)
optimizer = optim.Adam(net.parameters(), lr=l_rate)

'''
4、开始训练网络
'''
MAE_te_best = 10.0
te_outputs_best = np.zeros((te_size,4760))
outlog = open('../res/dense/dense1-'+file_num+'.log', 'w')
log_str = '\t'.join(map(str, ['epoch', 'MAE_va', 'MAE_te', 'MAE_tr',  'time(sec)']))
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
            log_str = '\t'.join(
                map(str, [(epoch * 18) + i + 1, '%.6f' % MAE_va, '%.6f' % MAE_te, '%.6f' % MAE_tr, int(t_new - t_old)]))
            print(log_str)
            outlog.write(log_str + '\n')
            sys.stdout.flush()
            # 保留最优MAE_te
            if MAE_te < MAE_te_best:
                MAE_te_best = MAE_te
                te_outputs_best = te_outputs.detach().cpu().numpy()
    print("epoch %d training over" % epoch)
print('MAE_te_best : %.6f' % (MAE_te_best))
np.save('../res/dense/dense1-'+file_num+'.npy', te_outputs_best)
outlog.write('MAE_te_best : %.6f' % (MAE_te_best) + '\n')
outlog.close()
print('Finish Training')

