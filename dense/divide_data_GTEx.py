import numpy as np

X_GTEx = np.load("../dataset/GTEx_X_float64.npy")
Y_GTEx = np.load("../dataset/GTEx_Y_float64.npy")

print(X_GTEx.shape)
print(Y_GTEx.shape)