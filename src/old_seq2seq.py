import torch as th
from data import *
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from dataloader import *
from utils import *
from copy import deepcopy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_size):
        super(Encoder, self).__init__()
        self.embedding_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # Use pretrain word vectors
        #if word_emb:
        #    self.embedding.weight = nn.Parameter(th.from_numpy(word_emb.vectors).float()) 
        self.RNN = nn.LSTM(self.embedding_size, self.hidden_size, bidirectional = True, batch_first = True)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.3)

    def forward(self, x, lengths):
        
        emb = self.embedding(x) # (B, S, 300)
        emb = self.drop(emb)
        packed = pack_padded_sequence(emb, lengths, batch_first = True)
        output, (h,c) = self.RNN(packed)
        output, _ = pad_packed_sequence(output, batch_first = True)

        h = h.permute(1,0,2).contiguous().view(-1, self.hidden_size*2)
        c = c.permute(1,0,2).contiguous().view(-1, self.hidden_size*2)
        #h = self.tanh(h)

        return output, (h,c)


class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, emb_size, attn = 'bilinear', pos = 'post'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.hidden_size = hidden_size
        self.RNNCell = nn.LSTMCell(emb_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, vocab_size)
        self.output_attn_fc = nn.Linear(hidden_size * 2, vocab_size)
        self.drop = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.pos = pos
        # Bilinear
        self.W = nn.Parameter(th.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.W)
        # Badanau
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Parameter(th.randn(hidden_size, 1))
        self.attention_style = attn
        # for pre attention
        self.pre_attn_fc = nn.Linear(hidden_size * 2, hidden_size)
        

    def forward(self, x, h_t, encoder_output = None, length = None):
        # x gets the shape (B, E) by passing the embedding layer of encoder
        x = self.embedding(x).squeeze(1)
        x = self.drop(x)
        # pre RNN attention
        h, c = h_t
        if encoder_output is not None and self.pos == 'pre':
            S = encoder_output.size(1)
            # Bilinear
            if self.attention_style == 'bilinear':
                energy = h.mm(self.W).unsqueeze(1).bmm(encoder_output.transpose(1,2)).squeeze(1) # (B, 1, H) x (B, H, S)
            # Dot product
            if self.attention_style == 'dot':
                energy = h.unsqueeze(1).bmm(encoder_output.transpose(1,2)).squeeze(1) # (B, 1, H) x (B, H, S)
            # Concat
            if self.attention_style == 'concat':
                concat = th.cat((h.unsqueeze(1).repeat(1,S,1), encoder_output), 2)
                energy = self.tanh(self.attn(concat)).matmul(self.V).permute(0,2,1).squeeze(1) # (B, 1, S)
            # masking
            mask = self.get_mask(length)
            # get weight
            energy = F.softmax(energy * mask, dim=-1)  # (B, S)
            context = energy.unsqueeze(1).bmm(encoder_output).squeeze(1) # (B, 1, S) x (B, S, H)
            h = self.pre_attn_fc(th.cat((context, h), 1))
            h_t = (h, c)
        
        h, c = self.RNNCell(x, h_t)
        # Do attention
        if encoder_output is not None and self.pos == 'post':
            S = encoder_output.size(1)
            # Bilinear
            if self.attention_style == 'bilinear':
                energy = h.mm(self.W).unsqueeze(1).bmm(encoder_output.transpose(1,2)).squeeze(1) # (B, 1, H) x (B, H, S)
            # Dot product
            if self.attention_style == 'dot':
                energy = h.unsqueeze(1).bmm(encoder_output.transpose(1,2)).squeeze(1) # (B, 1, H) x (B, H, S)
            # Concat
            if self.attention_style == 'concat':
                concat = th.cat((h.unsqueeze(1).repeat(1,S,1), encoder_output), 2)
                energy = self.tanh(self.attn(concat)).matmul(self.V).permute(0,2,1).squeeze(1) # (B, 1, S)
            # masking
            mask = self.get_mask(length)
            # get weight
            energy = F.softmax(energy * mask, dim=-1)  # (B, S)
            context = energy.unsqueeze(1).bmm(encoder_output).squeeze(1) # (B, 1, S) x (B, S, H)
            output = self.output_attn_fc(th.cat((context, h), 1))
            #output = self.output_fc(context)
        else:
            output = self.output_fc(h)


        return output, (h, c)
    
    def get_mask(self, length):
        mask = [[1 if i < l else 0 for i in range(length[0])] for l in length]
        return th.FloatTensor(mask)

    


        
