# 輸出策略
import numpy as np

def _pack(x, tap):
    pre_data = np.zeros([x.shape[0], tap, x.shape[1]])
    pre_order = np.zeros([tap-1, x.shape[1]])
    data = np.concatenate((pre_order, x), axis = 0)
    for kkk in range(tap):
        pre_data[:, kkk, :] = data[kkk:x.shape[0] + kkk, :]
    return pre_data 

def _nor2(x):
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    train_norm = (x - mu) / std
    return train_norm, mu, std

def _denor2(std, mu, x):
    denorm = x * std + mu
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

def _stock(trend):
    hold = 0
    ACT = []
    HOLD = []
    leng = trend.shape[0]
    for i in range(leng):
        if i>=leng:
            break
        else:
            if trend[i]==1:
                # up
                if hold==0:
                    act = -1
                    hold = -1
                elif hold==1:
                    act = -1
                    hold = 0
                elif hold==-1:
                    act = 0
                    hold = -1
            elif trend[i]==-1:
                # down
                if hold==0:
                    act = 1
                    hold = 1
                elif hold==1:
                    act = 0
                    hold = 1
                elif hold==-1:
                    act = 1
                    hold = 0
            else:
                act = 0
        ACT.append(act)
        HOLD.append(hold)
    return(np.array(ACT), np.array(HOLD))