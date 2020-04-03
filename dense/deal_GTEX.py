import numpy as np
import sys

model = str(sys.argv[1])
hidden = str(sys.argv[2])
dr = str(sys.argv[3])
ds = str(sys.argv[4])

filename = model+"-"+hidden+"-"+dr+"-"+ds
res_0 = np.load("../res/dense/"+filename+"-0.npy")
res_1 = np.load("../res/dense/"+filename+"-1.npy")
res_2 = np.load("../res/dense/"+filename+"-2.npy")
res_3 = np.load("../res/dense/"+filename+"-3.npy")
res_4 = np.load("../res/dense/"+filename+"-4.npy")

Y_te = np.array(np.load('../third_dataset/GTEx_Y_te_0-4760_float64.npy'))


res_avg = (res_0 + res_1 + res_2 + res_3 + res_4)/5
pre_avg_mean = (res_avg - Y_te).mean(axis=0)


avg_MAE = np.abs(res_avg - Y_te).mean()
print(avg_MAE)
print("----")
mae0 = np.abs(res_0 - Y_te).mean()
mae1 = np.abs(res_1 - Y_te).mean()
mae2 = np.abs(res_2 - Y_te).mean()
mae3 = np.abs(res_3 - Y_te).mean()
mae4 = np.abs(res_4 - Y_te).mean()
print(mae0)
print("----")

mse0 = np.square(res_0 - Y_te).mean()
mse1 = np.square(res_1 - Y_te).mean()
mse2 = np.square(res_2 - Y_te).mean()
mse3 = np.square(res_3 - Y_te).mean()
mse4 = np.square(res_4 - Y_te).mean()

delta0 = np.exp(-400 * np.log(mse0 / mse0))
delta1 = np.exp(-400 * np.log(mse1 / mse0))
delta2 = np.exp(-400 * np.log(mse2 / mse0))
delta3 = np.exp(-400 * np.log(mse3 / mse0))
delta4 = np.exp(-400 * np.log(mse4 / mse0))

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
print("---")
tmp2 = (np.abs(res_0 - res_avg) +
       np.abs(res_1 - res_avg) +
       np.abs(res_2 - res_avg) +
       np.abs(res_3 - res_avg) +
       np.abs(res_4 - res_avg))/5
std_error2 = tmp2.mean()
print(std_error2)