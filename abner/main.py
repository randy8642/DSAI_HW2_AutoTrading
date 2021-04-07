import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
import argparse

from functions import _pack
from functions import _label

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


#%%
D_tra, L_tra = _label(Data)
print(D_tra)