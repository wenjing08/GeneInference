import numpy as np

X_GTEx = np.load("./original_dataset/GTEx_X_float64.npy")
Y_GTEx_0 = np.load("./original_dataset/GTEx_Y_0-4760_float64.npy")
Y_GTEx_1 = np.load("./original_dataset/GTEx_Y_4760-9520_float64.npy")
print(X_GTEx.shape)
print(Y_GTEx_0.shape)
print(Y_GTEx_1.shape)


NUM_ = 2121
NUM__ = 2521
X_tr = X_GTEx[0:NUM_, :]
X_va = X_GTEx[NUM_:NUM__, :]
X_te = X_GTEx[NUM__:, :]

Y_tr_0 = Y_GTEx_0[0:NUM_, :]
Y_va_0 = Y_GTEx_0[NUM_:NUM__, :]
Y_te_0 = Y_GTEx_0[NUM__:, :]
Y_tr_1 = Y_GTEx_1[0:NUM_, :]
Y_va_1 = Y_GTEx_1[NUM_:NUM__, :]
Y_te_1 = Y_GTEx_1[NUM__:, :]
print(X_tr.shape)
print(X_va.shape)
print(X_te.shape)
print(Y_tr_0.shape)
print(Y_va_0.shape)
print(Y_te_0.shape)
print(Y_tr_1.shape)
print(Y_va_1.shape)
print(Y_te_1.shape)


np.save("./third_dataset/GTEx_X_tr_float64.npy", X_tr)
np.save("./third_dataset/GTEx_X_va_float64.npy", X_va)
np.save("./third_dataset/GTEx_X_te_float64.npy", X_te)

np.save("./third_dataset/GTEx_Y_tr_0-4760_float64.npy", Y_tr_0)
np.save("./third_dataset/GTEx_Y_tr_4760-9520_float64.npy", Y_tr_1)
np.save("./third_dataset/GTEx_Y_va_0-4760_float64.npy", Y_va_0)
np.save("./third_dataset/GTEx_Y_va_4760-9520_float64.npy", Y_va_1)
np.save("./third_dataset/GTEx_Y_te_0-4760_float64.npy", Y_te_0)
np.save("./third_dataset/GTEx_Y_te_4760-9520_float64.npy", Y_te_1)