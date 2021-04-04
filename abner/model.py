#%% Import packages
import torch
import torch.nn as nn
import math
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def trans(x):
    T = x.size(1)
    B = torch.zeros_like(x)
    for i, j in zip(reversed(range(T)), range(T)):
        B[:,j,:] = x[:,i, :]
    return B

#%% Sturcture
class Cell(nn.Module):
    def __init__(self, in_sz, out_sz):
        super(Cell, self).__init__()
        self.L_in = nn.Linear(in_sz, out_sz)
        self.L_mem = nn.Linear(out_sz, out_sz)
        self.LN = nn.LayerNorm(out_sz)
    def forward(self, x, h0):
        bz = x.size(0)
        h0 = h0.view(bz, 1, -1)
        Wx = self.L_in(x)
        Wh = self.L_mem(h0)
        out_LN = self.LN(Wx + Wh)
        y = torch.tanh(out_LN)
        hn = y[:,-1,:].unsqueeze(0)
        return y, hn

class Layer(nn.Module):
    def __init__(self, cell, N):
        super(Layer, self).__init__()
        self.layers = clones(cell, N-1)
    def forward(self, x, h0):
        i=0
        for layer in self.layers:
            x, h0 = layer(x, h0)
            if i==0:
                hn = h0
            else:
                hn = torch.cat((hn, h0))
            i+=1
        return x, hn

class m01(nn.Module):
    def __init__(self, in_sz, out_sz, hid):
        super(m01, self).__init__()
        self.hid = hid
        self.out_sz = out_sz
        #====
        self.m_L1 = Cell(in_sz, out_sz)
        self.md = Layer(Cell(out_sz, out_sz), hid)
        #====
        out2_sz = int(out_sz * (hid+1))
        self.FC = nn.Sequential(
            nn.Linear(out2_sz, 64),
            nn.ReLU(),
            nn.Linear(64, out_sz-1)
            )  
    def forward(self, x):
        bz = x.size(0)
        h0 = torch.zeros([1, bz, self.out_sz], device=x.device)
        # md
        if self.hid>1:
            x01, hn01 = self.m_L1(x, h0)
            y1, hn02 = self.md(x01, hn01)
            hn1 = torch.cat((hn01, hn02))
        else:
            y1, hn1 = self.m_L1(x, h0)

        y = y1
        hn = hn1
            
        # FC
        inn = x.reshape(bz, -1)
        pred = self.FC(inn)
        return pred, hn

#%% Test
if __name__ == "__main__":
    IN = torch.randn(32,20,4)
    F =  m01(4, 20, hid=3)   
    Pred, Hn = F(IN)
    print('Pred >>', Pred.size())
    print('Hn >>', Hn.size())