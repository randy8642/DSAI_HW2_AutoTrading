import numpy as np

def _pack(x, tap):
    pre_data = np.zeros([x.shape[0], tap, x.shape[1]])
    pre_order = np.zeros([tap-1, x.shape[1]])
    data = np.concatenate((pre_order, x), axis = 0)
    for kkk in range(tap):
        pre_data[:, kkk, :] = data[kkk:x.shape[0] + kkk, :]
    return pre_data 

def _nor(x):
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    train_norm = (x - mu) / std
    return train_norm, mu, std

def _comp(rec, out):
    if rec>out:
        trend = -1
    elif rec<out:
        trend = 1
    else:
        trend = 0
    return trend

def _stock(trend, hold):

    '''
    | 持有數量 | 預測明天相對今天 | 採取動作 |
    |----------|------------------|----------|
    | 1        | 漲 (1)           | 賣 (-1)  |
    | 1        | 跌 (-1)          | 無 (0)   |
    | 0        | 漲 (1)           | 賣 (-1)  |
    | 0        | 跌 (-1)          | 買 (1)   |
    | -1       | 漲 (1)           | 無 (0)   |
    | -1       | 跌 (-1)          | 買 (1)   |
    '''

    if trend==1:
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
    elif trend==-1:
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
    return act, hold