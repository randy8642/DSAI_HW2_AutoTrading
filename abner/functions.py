# 輸出策略
import numpy as np

def _pack(x, tap):
    pre_data = np.zeros([x.shape[0], tap, x.shape[1]])
    pre_order = np.zeros([tap-1, x.shape[1]])
    data = np.concatenate((pre_order, x), axis = 0)
    for kkk in range(tap):
        pre_data[:, kkk, :] = data[kkk:x.shape[0] + kkk, :]
    return pre_data 

def _label(x):
    leng = x.shape[0]
    x_cut = x[:leng-1, :]
    label = np.zeros((leng-1))
    for i in range(leng):
        if (i+1)==leng:
            break
        else:
            if x[i,0]<x[i+1,0]:
                label[i] = 1
    label = np.expand_dims(label, axis=1)
    return x_cut, label