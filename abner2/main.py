import numpy as np
import pandas as pd
import os
import random
import argparse

import functions
import config
import model

import torch
import torch.optim as optim
import torch.nn as nn

#%% Args
parser = argparse.ArgumentParser()
parser.add_argument('-R','--training',
                   default='training.csv',
                   help='input training data file name')

parser.add_argument('-E','--testing',
                    default='testing.csv',
                    help='input testing data file name')

parser.add_argument('-O','--output',
                    default='output.csv',
                    help='output file name')

args = parser.parse_args()

#%% Load
P = '../data'
Data = np.array(pd.read_csv(os.path.join(P, args.training), header=None))
Val = np.array(pd.read_csv(os.path.join(P, args.testing), header=None))

print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(3)

#%% Split
D_tra, mu, std = functions._nor(Data[:-1, :])
L_tra = Data[1:, 0]
L_tra = np.expand_dims(L_tra, axis=1)

D_tes, _, _ = functions._nor(Val)
L_tes = np.zeros((D_tes.shape[0], 1))

D_tra_T, L_tra_T = functions._pack(D_tra, config.tap), functions._pack(L_tra, config.tap)
D_tes_T, L_tes_T = functions._pack(D_tes, config.tap), functions._pack(L_tes, config.tap)

# dataset
train_data = torch.from_numpy(D_tra_T).type(torch.FloatTensor)
train_label = torch.from_numpy(L_tra_T).type(torch.FloatTensor)
train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=True)

test_data = torch.from_numpy(D_tes_T).type(torch.FloatTensor)
test_label = torch.from_numpy(L_tes_T).type(torch.FloatTensor)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=1, shuffle=False)

#%% Parameters
Epoch = config.ep
single_model = model.m03(4, 16, config.tap, hid=config.hid, bid=config.bid)
# single_model = model.m04(4, 16, config.tap, hid=config.hid, bid=config.bid)
single_optim = optim.SGD(single_model.parameters(), lr=config.lr)
# single_optim = optim.Adam(single_model.parameters(), lr=config.lr)
loss_f = nn.L1Loss()

single_model.to(device)
loss_f.to(device)

#%% Training
LOSS = []
print('\n------Training------')
single_model.train()
for epoch in range(Epoch):
    for n, (Data, Label) in enumerate(train_dataloader):
        single_optim.zero_grad()
        data = Data
        valid = Label[:, -1, :]

        data = data.to(device)
        valid = valid.to(device)

        pred, _ = single_model(data)
        loss = loss_f(pred, valid)

        loss.backward()
        single_optim.step()
        
    with torch.no_grad():
        print('epoch[{}], loss:{:.4f}'.format(epoch+1, loss.item()))
        LOSS.append(loss.item())
LOSS = np.array(LOSS)
  
#%% Testing

print('\n------Testing------')
single_model.eval()
with torch.no_grad():
    out = Val[0,0]
    hold = 0
    act_tot = []
    hold_tot = []
    for n_ts, (Data_ts, Label_ts) in enumerate (test_dataloader):
        rec = out

        data = Data_ts
        data = data.to(device)
            
        out, _ = single_model(data)
        out = out.cpu().data.numpy()

        trend = functions._comp(rec, out)
        act, hold = functions._stock(trend, hold)
        act_tot.append(act)
        hold_tot.append(hold)
        if n_ts==0:
            pred_tes = out
        else:
            pred_tes = np.concatenate((pred_tes, out), axis=0)

# Val.
pred_tes_ny = pred_tes.squeeze()
pred_tes = torch.from_numpy(pred_tes_ny).type(torch.FloatTensor)
pred_tes = pred_tes.to(device)

val_tes_ny = Val[1:,0]
val_tes = torch.from_numpy(val_tes_ny).type(torch.FloatTensor)
val_tes = val_tes.to(device)
loss_tes = loss_f(pred_tes[:-1], val_tes)
print(loss_tes)

#%% Trend
Result = np.array(act_tot)[:-1]

print('act')
print(act_tot[:-1])
print('\nhold')
print(hold_tot[:-1])

#%% Save
diction = {"Value": Result}
select_df = pd.DataFrame(diction)
sf = args.output
select_df.to_csv(sf,index=0,header=0)

loss_diction = {"loss": LOSS}
select_loss_df = pd.DataFrame(loss_diction)
select_loss_df.to_csv('loss_plot.csv',index=0,header=0)

pred_diction = {"pred": pred_tes_ny, "val": Val[:,0]}
select_pred_df = pd.DataFrame(pred_diction)
select_pred_df.to_csv('pred_plot.csv',index=0,header=0)