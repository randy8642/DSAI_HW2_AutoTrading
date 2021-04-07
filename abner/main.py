import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
import argparse

from functions import _pack
from functions import _label
from model import m02

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

#%%
D_tra, L_tra = _label(Data)
D_tes, L_tes = _label(Val)
D_tra_T, L_tra_T = _pack(D_tra, 20), _pack(L_tra, 20)
D_tes_T, L_tes_T = _pack(D_tes, 20), _pack(L_tes, 20)

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
Epoch = 30
single_model = m02(4, 1024, 20, hid=5, bid=True)
single_optim = optim.Adam(single_model.parameters(), lr=1e-4)
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

        loss.backward()
        single_optim.step()
        
    with torch.no_grad():
        print('epoch[{}], loss:{:.4f}'.format(epoch+1, loss.item()))

'''      
#%% Real Testing
print('\n------Testing------')
single_model.eval()
with torch.no_grad():
    for n_ts, (Data_ts, Label_ts) in enumerate (single_test_dataloader):

        data = Data_ts
        data = data.to(device)
            
        out, _ = single_model(fr_data)
        out = out.cpu().data.numpy()

        if n_ts==0:
            pred_tes = out
        else:
            pred_tes = np.concatenate((pred_tes, out), axis=0)
            
'''
