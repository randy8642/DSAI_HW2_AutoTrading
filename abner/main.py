import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
import random
import argparse

import functions
import config
import model

#%% Args
parser = argparse.ArgumentParser()
parser.add_argument('-TRA','--training',
                   default='training.csv',
                   help='input training data file name')

parser.add_argument('-TES','--testing',
                    default='testing.csv',
                    help='input testing data file name')

parser.add_argument('-O','--output',
                    default='output.csv',
                    help='output file name')

args = parser.parse_args()

#%% Load
P = '../data'
Data = np.array(pd.read_csv(os.path.join(P, args.training),header=None))
Val = np.array(pd.read_csv(os.path.join(P, args.testing),header=None))

print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

#%% Split
Data_n = functions._nor(Data)
Val_n = functions._nor(Val)

D_tra, L_tra = functions._label(Data_n)
D_tes = functions._label2(Val_n)
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
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=32, shuffle=False)

#%% Parameters
Epoch = config.ep
single_model = model.m03(4, 64, config.tap, hid=config.hid, bid=config.bid)
single_optim = optim.Adam(single_model.parameters(), lr=config.lr)
loss_f = nn.MSELoss()

single_model.to(device)
loss_f.to(device)

#%% Training
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
  
#%% Testing
print('\n------Testing------')
single_model.eval()
with torch.no_grad():
    for n_ts, (Data_ts, Label_ts) in enumerate (test_dataloader):

        data = Data_ts
        data = data.to(device)
            
        out, _ = single_model(data)
        out = out.cpu().data.numpy()

        if n_ts==0:
            pred_tes = out
        else:
            pred_tes = np.concatenate((pred_tes, out), axis=0)

# Val.
pred_tes_ny = functions._denor(Val[1:,:], pred_tes.squeeze())
pred_tes = torch.from_numpy(pred_tes_ny).type(torch.FloatTensor)
pred_tes = pred_tes.to(device)

val_tes_ny = Val[:,-1]
val_tes = torch.from_numpy(val_tes_ny).type(torch.FloatTensor)
val_tes = val_tes.to(device)
MSE_tes = loss_f(pred_tes, val_tes)
print(MSE_tes)

#%% Trend
'''
trend = functions._trend(pred_tes_ny)
act, hold = functions._stock(trend)
'''
'''
ACT = []
HOLD = [] 

act = functions.stock()
for idx, n in enumerate(trend):
    act.trade(n)
    HOLD.append(act.hold)
    ACT.append(act.actions)

Result = np.array(ACT[-1])
# print(Result)
# print(np.array(HOLD))

#%% Save
diction = {"Value": Result}
select_df = pd.DataFrame(diction)
# sf = args.model + "_" + args.output
sf = args.output
select_df.to_csv(sf,index=0,header=0)
'''