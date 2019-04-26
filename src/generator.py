import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam

class Generator(nn.Module):
    def __init__(self):
        self.embedding_size = emb_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(self.embedding_size * 2, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, z, x, state):
        '''
          z: prior from standard normal (B, E)
          x: the embedding from previous timestep (B, E)
          state: (h, c) from LSTM
        '''
        
        inp = th.cat((x, z), dim = -1)
        out, state = self.rnn(inp, state)
        out = self.fc(out)

        return out, state
    
        
        
class Discriminator(nn.Module):
    def __init__(self, emb_size, kernel_sizes):
        self.embedding_size = emb_size
        self.out_channel = 50
        cnns = []
        for k in kernel_size:
            cnns.append(nn.Conv1d(in_channels=self.embedding_size, out_channels=self.out_channel, kernel_size=k))
        
        self.CNNs = nn.ModuleList(cnns)
        self.transform = nn.Linear(self.out_channel * len(kernel_sizes), 1)
        
    
    def forward(self, x):
        feats = []
        B, S, _ = x.size()
        x = x.permute(0,2,1) # (B, E, S)
        for i in range(len(self.CNNs)):
            feat = self.CNNs[i](x) # (B, out channel, S)
            feat = F.max_pool1d(feat, kernel_size=S).squeeze() # (B, out_channel)
            feats.append(feat)

        feats = th.cat(feats. dim=-1)
        logit = self.transform(feats)
        prob = F.sigmoid(logit)

        return prob
        





