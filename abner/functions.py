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
    label = x[1:, 0]
    label = np.expand_dims(label, axis=1)
    return x_cut, label

def _label2(x):
    leng = x.shape[0]
    label = x[:, 0]
    label = np.expand_dims(label, axis=1)
    return x, label    

def _nor(x):
  train_norm = (x - np.mean(x)) / (np.max(x) - np.min(x))
  return train_norm  

def _denor(tes, x):
  denorm = x*(np.max(tes) - np.min(tes)) + np.mean(tes)
  return denorm

def _trend(x):
    out = np.zeros_like(x)
    leng = x.shape[0]
    for i in range(leng):
        if (i+2)>leng:
            break
        else:
            if x[i]<x[i+1]:
                out[i] = 1
            elif x[i]>x[i+1]:
                out[i] = -1  
            else:
                out[i] = 0           
    return out[:-1]

class stock():
    def __init__(self) -> None:
        self.hold = 0

        self.actions = []
        pass


    def trade(self, predict) -> int:
        '''
        買賣交易
        ---
        input\\

        -1:跌
        0:持平
        1:漲

        option\\

        持有數量       | 今天收盤  | 預測明天收盤 | 動作 
        -------------- | :-----: | :-----: | :----:
        1    | 低 |  高 |    無 0
        1    | 高 |  低 |    賣 -1
        0    | 低 |  高 |    買 1
        0    | 高 |  低 |    賣 -1
        -1    | 低 |  高 |    買 1
        -1    | 高 |  低 |    無 0

        '''
        if self.hold == 1:
            if predict == 1:
                self.actions.append(0)
            elif predict == 0:
                self.actions.append(0)
            elif predict == -1:
                self.actions.append(-1)
        elif self.hold == 0:
            if predict == 1:
                self.actions.append(1)
            elif predict == 0:
                self.actions.append(0)
            elif predict == -1:
                self.actions.append(-1)
        elif self.hold == -1:
            if predict == 1:
                self.actions.append(1)
            elif predict == 0:
                self.actions.append(0)
            elif predict == -1:
                self.actions.append(0)

        # 
        self.hold += self.actions[-1]

        return self.actions[-1]