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
            if x[i,-1]<x[i+1,-1]:
                label[i] = 1
    label = np.expand_dims(label, axis=1)
    return x_cut, label

def _label2(x):
    leng = x.shape[0]
    x_cut = x[:leng, :]
    label = x[:, -1]
    label = np.expand_dims(label, axis=1)
    return x_cut, label

def _MAV(x, tap):
    y = np.zeros_like(x)
    leng = x.shape[0]
    for i in range(leng):
        if (i+1)<tap:
            y[i,:] = np.mean(x[:i+1, :], axis=0)
        else:
            y[i,:] = np.mean(x[i-tap+1:i, :], axis=0)
    return y

def _nor(x):
  train_norm = (x - np.mean(x)) / (np.max(x) - np.min(x))
  return train_norm  

def _denor(tes, x):
  denorm = x*(np.max(tes) - np.min(tes)) + np.mean(tes)
  return denorm
