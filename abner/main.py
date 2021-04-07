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


parser = argparse.ArgumentParser()
parser.add_argument('-TRA','--training',
                   default='training.csv',
                   help='input training data file name')

parser.add_argument('-TES','--testing',
                    default='testing.csv',
                    help='input testing data file name')

parser.add_argument('-O','--output',
                    default='submission.csv',
                    help='output file name')

args = parser.parse_args()

#%% 
P = '../data'
Data = np.array(pd.read_csv(os.path.join(P, args.training)))
Val = np.array(pd.read_csv(os.path.join(P, args.testing)))

print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

#%%
D_tra, L_tra = functions._label(Data)
D_tes, L_tes = functions._label(Val)
D_tra_T, L_tra_T = functions._pack(D_tra, config.tap), functions._pack(L_tra, config.tap)
D_tes_T, L_tes_T = functions._pack(D_tes, config.tap), functions._pack(L_tes, config.tap)

D_tra_T = np.expand_dims(D_tra_T[:,:,-1], axis=-1)
D_tes_T = np.expand_dims(D_tes_T[:,:,-1], axis=-1)

#%%
train_data = torch.from_numpy(D_tra_T).type(torch.FloatTensor)
train_label = torch.from_numpy(L_tra_T).type(torch.FloatTensor)
train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=32, shuffle=True)

test_data = torch.from_numpy(D_tes_T).type(torch.FloatTensor)
test_label = torch.from_numpy(L_tes_T).type(torch.FloatTensor)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=32, shuffle=False)

#%%
Epoch = config.ep
single_model = model.m02(1, 1, config.tap, hid=config.hid, bid=config.bid)
# single_model = model.m03(4, 4, config.tap, hid=config.hid, bid=config.bid)
single_optim = optim.Adam(single_model.parameters(), lr=config.lr)
bce_loss = nn.BCELoss()

single_model.to(device)
bce_loss.to(device)

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
        loss = bce_loss(pred, valid)
        A = (pred.round()==valid)
        acc = torch.sum(A)/data.size(0)

        loss.backward()
        single_optim.step()
        
    with torch.no_grad():
        print('epoch[{}], loss:{:.4f}, acc:{:.4f}'.format(epoch+1, loss.item(), acc.item()))

  
#%% Real Testing
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
            

A = (pred_tes.round()==L_tes)
acc = np.sum(A)/L_tes_T.shape[0]
print('Accuracy >>', round(acc, 5))
