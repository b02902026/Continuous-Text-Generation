import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam

class Encoder(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size):
        super(Encoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        # define layers
        self.embedding = nn.Embedding(vocab_size, self.emb_size)
        self.encoder = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.h_transform = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.c_transform = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.drop = nn.Dropout(0.3)
        

    def forward(self, x, lengths):
        
        embed = self.embedding(x) # (B, S, E)
        packed = pack_padded_sequence(embed, lengths, batch_first=True)
        output, (h, c) = self.encoder(packed) # (B, S, H), (n_layer*direction, B, H)
        h = h.permute(1,0,2).contiguous().view(-1, self.hidden_size * 2)
        c = c.permute(1,0,2).contiguous().view(-1, self.hidden_size * 2)
        h = self.h_transform(h)
        c = self.c_transform(c)

        return h, c

class Decoder(nn.Module):
    def __init__(self, emb_size, embedding_layer, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding_layer
        self.emb_size = emb_size
        self.decoder = nn.LSTMCell(self.emb_size, self.hidden_size)
        self.output_fc = nn.Linear(self.hidden_size, vocab_size)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.3)
    
    def forward(self, src, state):
        
        emb = self.embedding(src).squeeze(1)
        h, c = self.decoder(emb, state)
        prob = self.output_fc(h)
        #prob = F.log_softmax(prob, dim=-1)

        return prob, (h, c)


        
class Discriminator(nn.Module):
    def __init__(self, emb_size, kernel_sizes):
        super(Discriminator, self).__init__()
        self.embedding_size = emb_size
        self.out_channel = 50
        cnns = []
        for k in kernel_sizes:
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

        feats = th.cat(feats, dim=-1)
        logit = self.transform(feats)
        prob = F.sigmoid(logit)

        return prob
        





