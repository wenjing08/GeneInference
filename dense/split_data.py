import numpy as np

# Y_GTEx = np.load("../dataset/GTEx_Y_float64.npy")
# print(Y_GTEx.shape)
# Y_GTEx_4760 = Y_GTEx[:,0:4760]
# Y_GTEx_9520 = Y_GTEx[:, 4760:9520]
# np.save("../dataset/GTEx_Y_4760_float64.npy",Y_GTEx_4760)
# np.save("../dataset/GTEx_Y_9520-2_float64.npy",Y_GTEx_9520)
# exit()


# Y_tr = np.load("../dataset/bgedv2_Y_tr_float64.npy")
# print(Y_tr.shape)
# Y_tr_4760 = Y_tr[:, 0:4760]
# Y_tr_9520 = Y_tr[:, 4760:9520]
# print(Y_tr_9520.shape)
# np.save("../dataset/bgedv2_Y_tr_4760_float64.npy", Y_tr_4760)
# np.save("../dataset/bgedv2_Y_tr_9520-2_float64.npy", Y_tr_9520)
#
# Y_va = np.load("../dataset/bgedv2_Y_va_float64.npy")
# print(Y_va.shape)
# Y_va_4760 = Y_va[:, 0:4760]
# Y_va_9520 = Y_va[:, 4760:9520]
# print(Y_va_9520.shape)
# np.save("../dataset/bgedv2_Y_va_4760_float64.npy", Y_va_4760)
# np.save("../dataset/bgedv2_Y_va_9520-2_float64.npy", Y_va_9520)
#
# Y_te = np.load("../dataset/bgedv2_Y_te_float64.npy")
# print(Y_te.shape)
# Y_te_4760 = Y_te[:, 0:4760]
# Y_te_9520 = Y_te[:, 4760:9520]
# print(Y_te_9520.shape)
# np.save("../dataset/bgedv2_Y_te_4760_float64.npy", Y_te_4760)
# np.save("../dataset/bgedv2_Y_te_9520-2_float64.npy", Y_te_9520)

# Y_tr = np.load("../dataset/bgedv2_Y_tr_float64.npy")
# print(Y_tr.shape)
# Y_tr_1000 = Y_tr[:, 0:1000]
# print(Y_tr_1000.shape)
# np.save("../dataset/bgedv2_Y_tr_1000_float64.npy", Y_tr_1000)
#
# Y_va = np.load("../dataset/bgedv2_Y_va_float64.npy")
# print(Y_va.shape)
# Y_va_1000 = Y_va[:, 0:1000]
# print(Y_va_1000.shape)
# np.save("../dataset/bgedv2_Y_va_1000_float64.npy", Y_va_1000)
#
# Y_te = np.load("../dataset/bgedv2_Y_te_float64.npy")
# print(Y_te.shape)
# Y_te_1000 = Y_te[:, 0:1000]
# print(Y_te_1000.shape)
# np.save("../dataset/bgedv2_Y_te_1000_float64.npy", Y_te_1000)
#
# Y_GTEx = np.load("../dataset/GTEx_Y_float64.npy")
# print(Y_GTEx.shape)
# Y_GTEx_1000 = Y_GTEx[:,0:1000]
# print(Y_GTEx_1000.shape)
# np.save("../dataset/GTEx_Y_1000_float64.npy",Y_GTEx_1000)
begin = 15
end = 20
num = 3

Y_tr = np.load("../dataset/bgedv2_Y_tr_float64.npy")
print(Y_tr.shape)
Y_tr_5 = Y_tr[:, begin:end]
print(Y_tr_5.shape)
np.save("../dataset/bgedv2_Y_tr_5_"+str(num)+"_float64.npy", Y_tr_5)

Y_va = np.load("../dataset/bgedv2_Y_va_float64.npy")
print(Y_va.shape)
Y_va_5 = Y_va[:, begin:end]
print(Y_va_5.shape)
np.save("../dataset/bgedv2_Y_va_5_"+str(num)+"_float64.npy", Y_va_5)

Y_te = np.load("../dataset/bgedv2_Y_te_float64.npy")
print(Y_te.shape)
Y_te_5 = Y_te[:, begin:end]
print(Y_te_5.shape)
np.save("../dataset/bgedv2_Y_te_5_"+str(num)+"_float64.npy", Y_te_5)

Y_GTEx = np.load("../dataset/GTEx_Y_float64.npy")
print(Y_GTEx.shape)
Y_GTEx_5 = Y_GTEx[:, begin:end]
print(Y_GTEx_5.shape)
np.save("../dataset/GTEx_Y_5_"+str(num)+"_float64.npy", Y_GTEx_5)