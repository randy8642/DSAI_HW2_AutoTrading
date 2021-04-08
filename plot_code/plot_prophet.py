import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df_train = pd.read_csv('./data/training.csv', header=None)
df_train.columns = ['Open', 'High', 'Low', 'Close']

df = df_train[['Open']]
df = df.rename({'Open': 'y'}, axis='columns')
df['ds'] = pd.date_range('1-1-2017', periods=len(df_train), freq='D')

print(df)

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=20)
# future.tail()

forecast = m.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

print(forecast)
forecast.to_csv('forcast.csv',encoding='utf-8')