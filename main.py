from sys import prefix
import pytorch_lightning as pl
import torch.nn as nn
import torch.utils.data as data
from typing import Optional
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# MODEL
class lightmodel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=20,
                          num_layers=1, batch_first=True)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)

        pred = self.layers(out)
        return pred

    def training_step(self, train_batch, batch_idx):

        input, label = train_batch

        out, _ = self.gru(input)

        pred = self.layers(out)

        loss = nn.functional.mse_loss(pred, label)

        return loss

    def configure_optimizers(self):
        '''
        optim.
        '''
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

# DATA


class dataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        df_train = pd.read_csv('./data/training.csv', header=None)
        df_train.columns = ['Open', 'High', 'Low', 'Close']

        df = KD(df_train)

        df = df[['Open', 'K', 'D']]

        data = df.to_numpy()
        data = np.nan_to_num(data)
        data = torch.from_numpy(data)

        x = torch.zeros([0, 20, 3])
        label = torch.zeros([0, 1])
        for i in range(data.shape[0]-20):
            d = data[i:i+20, :].reshape(1, 20, 3)
            x = torch.cat((x, d), dim=0)
            l = data[i+20, 0].reshape(1, 1)
            label = torch.cat((label, l), dim=0)

        x = x.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        self.train_data = list(zip(x, label))

    def train_dataloader(self):
        train_loader = data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=False)
        return train_loader

    # def test_dataloader(self):
    #     test_loader = data.DataLoader(
    #         self.test_data, batch_size=self.batch_size, shuffle=False)
    #     return test_loader


model = lightmodel()
trainer = pl.Trainer(max_epochs=100, logger=False, checkpoint_callback=False)
dm = dataModule(batch_size=64)

trainer.fit(model, dm)


# TEST
df_train = pd.read_csv('./data/training.csv', header=None)
df_train.columns = ['Open', 'High', 'Low', 'Close']
df_test = pd.read_csv('./data/testing.csv', header=None)
df_test.columns = ['Open', 'High', 'Low', 'Close']

df = pd.concat([df_train, df_test], ignore_index=True)
df = KD(df)
df = df[['Open', 'K', 'D']]

stock_data = df.to_numpy()
stock_data = np.nan_to_num(stock_data)


stock_data = torch.from_numpy(stock_data)
x = torch.zeros([0, 20, 3])
label = torch.zeros([0, 1])
for i in range(len(df_train.index)-20, len(stock_data)-20, 1):
    d = stock_data[i:i+20, :].reshape(1, 20, 3)
    x = torch.cat((x, d), dim=0)
    l = stock_data[i+20, 0].reshape(1, 1)
    label = torch.cat((label, l), dim=0)

x = x.type(torch.FloatTensor)
label = label.type(torch.FloatTensor)

real = []
pred = []

for day in range(20):
    out = model(x[day:day+1, :, :])
    pred.append(out.detach().numpy()[0, 0])
    real.append(label.numpy()[day, 0])

print(real)
print(pred)

plt.plot(real)
plt.plot(pred)


plt.show()
