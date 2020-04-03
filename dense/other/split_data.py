import numpy as np


Y_tr = np.load("../dataset/bgedv2_Y_tr_float64.npy")
print(Y_tr.shape)
Y_tr_6346 = Y_tr[:, 3173:6346]
print(Y_tr_6346.shape)
np.save("../dataset/bgedv2_Y_tr_6346_float64.npy", Y_tr_6346)


Y_va = np.load("../dataset/bgedv2_Y_va_float64.npy")
print(Y_va.shape)
Y_va_6346 = Y_va[:, 3173:6346]
print(Y_va_6346.shape)
np.save("../dataset/bgedv2_Y_va_6346_float64.npy", Y_va_6346)


Y_te = np.load("../dataset/bgedv2_Y_te_float64.npy")
print(Y_te.shape)
Y_te_6346 = Y_te[:, 3173:6346]
print(Y_te_6346.shape)
np.save("../dataset/bgedv2_Y_te_6346_float64.npy", Y_te_6346)