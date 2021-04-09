import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

import functions

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman'

P = '../other_csv'
sP = '../img'

#%% Data
Data_loss = np.array(pd.read_csv(os.path.join(P, 'loss_plot.csv'), header=None))
Pred_loss = np.array(pd.read_csv(os.path.join(P, 'pred_plot.csv'), header=None))

#%% Loss
fig, ax = plt.subplots(1, 1, figsize = (10,5))

ep = np.linspace(1,20,20)
# ep = np.linspace(1,60,60)
yticksL = [0, 30, 60, 90, 120, 150]
# xticksL = [1, 10, 20, 30, 40, 50, 60]
xticksL = [1, 5, 10, 15, 20]

plt.title("Loss Figure", fontsize=30)
ax.plot(ep, Data_loss)
ax.set_xlim(ep[0], ep[-1])
ax.set_xlabel('Epoch', size=25)
ax.set_xticks(xticksL)
ax.set_xticklabels(xticksL, fontsize=20)
# ax.set_xticks(Ses)
ax.set_ylabel('L1 Loss', size=25)
ax.set_yticks(yticksL)
ax.set_yticklabels(yticksL, fontsize=20)
plt.tight_layout()

plt.savefig(os.path.join(sP, 'loss_cv.png'))
# plt.show()

#%% Pred
fig, ax = plt.subplots(1, 1, figsize = (10,5))

x = np.linspace(1,19,19)
pred, _, _ = functions._nor(Pred_loss[1:,0])
val, _, _ = functions._nor(Pred_loss[1:,1])
yticksL = [-2, -1, 0, 1, 2]
xticksL = [1, 5, 10, 15, 19]

plt.title("Compare Trend", fontsize=30)
ax.plot(x, pred, label='Pred.')
ax.plot(x, val, label='Val.')
ax.set_xlim(x[0], x[-1])
ax.set_xlabel('Day', size=25)
ax.set_xticks(np.int_(x))
ax.set_xticklabels(np.int_(x), fontsize=20)
# ax.set_xticks(Ses)
ax.set_ylabel('Trend', size=25)
ax.set_yticks(yticksL)
ax.set_yticklabels(yticksL, fontsize=20)
plt.legend(loc=1, fontsize=20)
plt.tight_layout()

plt.savefig(os.path.join(sP, 'trend_cv.png'))
# plt.show()