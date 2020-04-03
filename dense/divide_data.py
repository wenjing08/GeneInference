import numpy as np

X_tr = np.load("../dataset/bgedv2_X_tr_float64.npy")
Y_tr = np.load("../dataset/bgedv2_Y_tr_float64.npy")

tr = np.hstack((X_tr, Y_tr))
# print(tr.shape)

X_va = np.load("../dataset/bgedv2_X_va_float64.npy")
Y_va = np.load("../dataset/bgedv2_Y_va_float64.npy")

va = np.hstack((X_va, Y_va))
# print(va.shape)

X_te = np.load("../dataset/bgedv2_X_te_float64.npy")
Y_te = np.load("../dataset/bgedv2_Y_te_float64.npy")

te = np.hstack((X_te, Y_te))
# print(te.shape)

tmp = np.vstack((tr, va))
all_data = np.vstack((tmp, te))
# print(all_data.shape)

np.random.shuffle(all_data)
# print(all_data.shape)

# 已打乱 直接划分就好

tr_new = all_data[0: 77706, :]
va_new = all_data[77706:88807, :]
te_new = all_data[88807:, :]
print(tr_new.shape)
print(va_new.shape)
print(te_new.shape)

x_tr_new = tr_new[:,0:943]
y_tr_0_4760_new = tr_new[:, 943:5703]
y_tr_4760_9520_new = tr_new[:, 5703:]
print(x_tr_new.shape)
print(y_tr_0_4760_new.shape)
print(y_tr_4760_9520_new.shape)
np.save("../dataset/bgedv2_X_tr_new_float64.npy", x_tr_new)
np.save("../dataset/bgedv2_Y_tr_new_4760_float64.npy", y_tr_0_4760_new)
np.save("../dataset/bgedv2_Y_tr_new_9520-2_float64.npy", y_tr_4760_9520_new)

x_va_new = va_new[:,0:943]
y_va_0_4760_new = va_new[:, 943:5703]
y_va_4760_9520_new = va_new[:, 5703:]
print(x_va_new.shape)
print(y_va_0_4760_new.shape)
print(y_va_4760_9520_new.shape)
np.save("../dataset/bgedv2_X_va_new_float64.npy", x_va_new)
np.save("../dataset/bgedv2_Y_va_new_4760_float64.npy", y_va_0_4760_new)
np.save("../dataset/bgedv2_Y_va_new_9520-2_float64.npy", y_va_4760_9520_new)

x_te_new = te_new[:,0:943]
y_te_0_4760_new = te_new[:, 943:5703]
y_te_4760_9520_new = te_new[:, 5703:]
print(x_te_new.shape)
print(y_te_0_4760_new.shape)
print(y_te_4760_9520_new.shape)
np.save("../dataset/bgedv2_X_te_new_float64.npy", x_te_new)
np.save("../dataset/bgedv2_Y_te_new_4760_float64.npy", y_te_0_4760_new)
np.save("../dataset/bgedv2_Y_te_new_9520-2_float64.npy", y_te_4760_9520_new)