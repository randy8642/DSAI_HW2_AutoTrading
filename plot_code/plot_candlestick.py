import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc

df_train = pd.read_csv('./data/training.csv', header=None)
df_train.columns = ['Open', 'High', 'Low', 'Close']

# open-high-low-close

############################################################################
fig, ax = plt.subplots(2, 1, sharex=True)

# CANDLESTICK
ohlc = df_train
ohlc['Date'] = range(len(ohlc.index))
ohlc = ohlc.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc = ohlc[-180:]

candlestick_ohlc(ax[0], ohlc.values, width=0.8,
                 colorup='green', colordown='red', alpha=0.8)
ax[0].set_xticklabels([])

# KD


def KD(data):
    data_df = data.copy()
    data_df['min'] = data_df['Low'].rolling(9).min()
    data_df['max'] = data_df['High'].rolling(9).max()
    data_df['RSV'] = (data_df['Close'] - data_df['min']) / \
        (data_df['max'] - data_df['min'])
    data_df = data_df.dropna()
    # 計算K
    # K的初始值定為50
    K_list = [50]
    for num, rsv in enumerate(list(data_df['RSV'])):
        K_yestarday = K_list[num]
        K_today = 2/3 * K_yestarday + 1/3 * rsv
        K_list.append(K_today)
    data_df['K'] = K_list[1:]
    # 計算D
    # D的初始值定為50
    D_list = [50]
    for num, K in enumerate(list(data_df['K'])):
        D_yestarday = D_list[num]
        D_today = 2/3 * D_yestarday + 1/3 * K
        D_list.append(D_today)
    data_df['D'] = D_list[1:]
    use_df = pd.merge(data, data_df[['K', 'D']],
                      left_index=True, right_index=True, how='left')
    return use_df


d = KD(df_train)
d = d[-180:]

ax[1].plot(d['K'], label='K')
ax[1].plot(d['D'], label='D')
ax[1].legend(loc=1)
ax[1].set_xticklabels([])


fig.suptitle('training last 180 days')
plt.show()
