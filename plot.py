import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates

df_train = pd.read_csv('./data/training.csv', header=None)
df_train.columns = ['Open', 'High', 'Low', 'Close']
df_test = pd.read_csv('./data/testing.csv', header=None)
# open-high-low-close

ohlc = df_train
ohlc['Date'] = range(len(ohlc.index))
ohlc = ohlc.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]

print(ohlc)

fig,ax = plt.subplots(1,1)




candlestick_ohlc(ax, ohlc.values[:350], width=0.6, colorup='green', colordown='red', alpha=0.8)


plt.show()
