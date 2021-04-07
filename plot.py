import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_train = pd.read_csv('./data/training.csv', header=None)
df_test = pd.read_csv('./data/testing.csv', header=None)
# open-high-low-close
# train = df_train[3].to_numpy()
# test = df_test[3].to_numpy()

# x1 = np.arange(0, train.shape[0], 1)
# x2 = np.arange(train.shape[0], train.shape[0]+test.shape[0], 1)

# plt.plot(x1, train, label='train')
# plt.plot(x2, test, label='test')
# plt.legend()
# plt.xticks(ticks=[],labels=[])
# plt.grid()
# plt.show()

plt.rc('figure',figsize=(16,12))
plt.rc('font',size=13)
from statsmodels.tsa.seasonal import STL
co2 = pd.Series(df_train[3].to_list(), index=pd.date_range('1-1-2017', periods=len(df_train[3]), freq='D'), name = 'stock')
print(co2.describe())
stl = STL(co2, seasonal=7)
res = stl.fit()
print(res.trend)
plt.plot(df_train[3].to_list() -  res.trend)
plt.show()
fig = res.plot()
plt.show()




