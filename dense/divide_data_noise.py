import numpy as np
import random
Y_tr_0 = np.load("../original_dataset/bgedv2_Y_tr_0-4760_float64.npy")
Y_tr_1 = np.load("../original_dataset/bgedv2_Y_tr_4760-9520_float64.npy")

list = random.sample(range(0,88807), 900)

print(Y_tr_0[list].shape)
print(Y_tr_1[list].shape)

Y_tr_0[list] = Y_tr_0[list] + np.random.normal(loc=0, scale=10, size=(900, 4760))
Y_tr_1[list] = Y_tr_1[list] + np.random.normal(loc=0, scale=10, size=(900, 4760))

np.save("../noise_dataset/bgedv2_Y_tr_0-4760_float64.npy", Y_tr_0)
np.save("../noise_dataset/bgedv2_Y_tr_4760-9520_float64.npy", Y_tr_1)
print("over")