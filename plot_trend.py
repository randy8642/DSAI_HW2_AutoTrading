import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL

plt.rc('figure',figsize=(16,12))
plt.rc('font',size=13)


df_train = pd.read_csv('./data/training.csv', header=None)
df_train.columns = ['Open', 'High', 'Low', 'Close']

co2 = pd.Series(df_train['Open'].to_list(), index=pd.date_range('1-1-2017', periods=len(df_train), freq='D'), name = 'open')
print(co2)

stl = STL(co2, seasonal=13)
res = stl.fit()
fig = res.plot()

plt.show()