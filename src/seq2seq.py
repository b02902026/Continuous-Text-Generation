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
        

    def forward(self, x, lengths):
        
        embed = self.embedding(x) # (B, S, E)
        packed = pack_padded_sequence(embed, lengths, batch_first=True)
        output, (h, c) = self.encoder(packed) # (B, S, H), (n_layer*direction, B, H)
        h = h.permute(1,0,2).contiguous().view(-1, self.hidden_size * 2)
        c = c.permute(1,0,2).contiguous().view(-1, self.hidden_size * 2)
        
        return h, c

class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = embedding_layer
        self.emb_size = emb_size
        self.decoder = nn.LSTMCell(self.emb_size, self.hidden_size)
        self.output_fc = nn.Linear(self.hidden_size, vocab_size)
        self.embedding = nn.Linear(self.hidden_size, self.emb_size)
    
    def forward(self, emb, state):
        
        h, c = self.decoder(emb, state)
        prob = self.output_fc(h)
        vec = self.embedding(h)

        return vec, prob, (h, c)
        
    
    
        
        
