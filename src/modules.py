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
        self.drop = nn.Dropout(0.2)
        self.embedding_matrix = nn.Parameter(th.FloatTensor(th.randn(vocab_size, self.emb_size)))
        

    def forward(self, x, lengths):
        
        #embed = self.embedding(x) # (B, S, E)
        embed = x.matmul(self.embedding_matrix) # (B, S, V) * (V, E)
        packed = pack_padded_sequence(embed, lengths, batch_first=True)
        output, (h, c) = self.encoder(packed) # (B, S, H), (n_layer*direction, B, H)
        h = h.permute(1,0,2).contiguous().view(-1, self.hidden_size * 2)
        c = c.permute(1,0,2).contiguous().view(-1, self.hidden_size * 2)
        h = self.h_transform(h)
        c = self.c_transform(c)

        return h, c

class Decoder(nn.Module):
    def __init__(self, emb_size, embedding_matrix, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        #self.decoder = nn.LSTMCell(self.emb_size, self.hidden_size)
        self.output_fc = nn.Linear(self.hidden_size, vocab_size)
        self.tanh = nn.Tanh()
        self.decoder = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.embedding_matrix = embedding_matrix
    
    def forward(self, src, state):
        
        #emb = self.embedding(src)
        emb = src.matmul(self.embedding_matrix)
        output, (h, c) = self.decoder(emb, state)
        prob = self.output_fc(output)

        return prob, (h, c)


        
class Discriminator(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.f = nn.GRU(emb_size, hidden_size, batch_first = True, bidirectional=True) 
        self.transform = nn.Linear(self.hidden_size*2, 1)

    def forward(self, x, lengths):
        B, S, _ = x.size()
        lengths_sort, idx_sort = th.sort(lengths, descending=True)
        x_sort = th.index_select(x, 0, idx_sort)
        packed = pack_padded_sequence(x_sort, lengths_sort, batch_first=True)
        output, h = self.f(packed)
        h = h.permute(1,0,2).contiguous().view(B, -1)
        logit = self.transform(h)
        prob = th.sigmoid(logit)

        return prob
        





