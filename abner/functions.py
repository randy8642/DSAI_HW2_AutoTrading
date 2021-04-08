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

def _stock(trend):
    hold = 0
    ACT = []
    HOLD = []
    leng = trend.shape[0]

    '''
    買賣交易
    ---
    input\\

    -1:跌
    0:持平
    1:漲

    option\\

    | 持有數量 | 預測明天相對今天 | 採取動作 |
    |----------|------------------|----------|
    | 1        | 漲 (1)           | 賣 (-1)  |
    | 1        | 跌 (-1)          | 無 (0)   |
    | 0        | 漲 (1)           | 賣 (-1)  |
    | 0        | 跌 (-1)          | 買 (1)   |
    | -1       | 漲 (1)           | 無 (0)   |
    | -1       | 跌 (-1)          | 買 (1)   |

    '''

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


def _stock2(trend):
    hold = 0
    ACT = []
    HOLD = []
    leng = trend.shape[0]

    '''
    買賣交易
    ---
    input\\

    -1:跌
    0:持平
    1:漲

    option\\

    | 持有數量 | 預測後兩天 | 採取動作 |
    |----------|------------|----------|
    | 1        | -1 -1      | 無 (0)   |
    | 1        | -1 +1      | 無 (0)   |
    | 1        | +1 -1      | 賣 (-1)  |
    | 1        | +1 +1      | 無 (0)   |
    | 0        | -1 -1      | 無 (0)   |
    | 0        | -1 +1      | 買 (1)   |
    | 0        | +1 -1      | 賣 (-1)  |
    | 0        | +1 +1      | 無 (0)   |
    | -1       | -1 -1      | 無 (0)   |
    | -1       | -1 +1      | 買 (1)   |
    | -1       | +1 -1      | 無 (0)   |
    | -1       | +1 +1      | 無 (0)   |

    '''

    for i in range(leng):
        if i+2>=leng:
            break
        else:
            if trend[i]==-1 and trend[i+1]==-1:
                # --
                act = 0
            elif trend[i]==-1 and trend[i+1]==1:
                # -+
                if hold==0:
                    act = 1
                    hold = 1
                elif hold==1:
                    act = 0
                    hold = 1
                elif hold==-1:
                    act = 1
                    hold = 0
            elif trend[i]==1 and trend[i+1]==-1:
                # +-
                if hold==0:
                    act = -1
                    hold = 0
                elif hold==1:
                    act = -1
                    hold = 0
                elif hold==-1:
                    act = 0
                    hold = -1 
            elif trend[i]==1 and trend[i+1]==1:
                # ++
                act = 0                                       
            else:
                act = 0
        ACT.append(act)
        HOLD.append(hold)

    for j in range(2):
        ACT.append(0)
        HOLD.append(hold)

    return(np.array(ACT), np.array(HOLD))    