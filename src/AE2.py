import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from vs_autoencoder import Autoencoder

class SentenceAE(nn.Module):
    def __init__(self, vocab, loss_fn):
        self.hidden_size = 200
        self.embedding_size = 300
        self.vocab_size = len(vocab)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.encoder = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.decoder = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.vocab = vocab
    
    def encode(self, src, lengths):
        
        emb = self.embedding(src)
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        out, h = self.encoder(packed)
        h = h.contiguous().view(1, -1, self.hidden_size*2)

        return out, h
    
    def decode(self, src, hidden):
        
        gt

        


    def train(self, dataloader, training=True, device='cpu'):

        total_loss = 0
        aug_loss_fn = nn.CosineEmbeddingLoss(reduction='sum') 
        for i, (x, lengths) in enumerate(dataloader):
            loss = 0
            aug_loss = 0
            self.optim.zero_grad()
            B, S = x.size()
            state = self.encoder(x, lengths) # (B, 2H)
            inp = th.LongTensor([[self.vocab("<bos>")]]*B).view(B, 1).to(device) # (B, 1)
            inp = self.encoder.embedding(inp).view(B, -1) # (B, E)
            for t in range(S):
                vec , logit, state = self.decoder(inp, state)
                loss += self.loss_fn(logit, x[:,t])
                # Normal seq2seq
                inp = self.encoder.embedding(x[:, t])
                # Training using embedding
                #inp = vec.view(B, -1)
                #aug_loss += aug_loss_fn(vec, self.encoder.embedding(x[:, t]).data, th.ones((B,1)))
            
            #loss += aug_loss
            loss /= lengths.sum().item()
            if training:
                loss.backward()
                self.optim.step()
            
            
            total_loss += loss.item()
            print("\rbatch:{}/{}, loss:{}, total loss:{}".format(i, len(dataloader), loss.item(), total_loss / (i + 1)), end='')
            
    
    def inference(self, src):
        
        S = src.size(0)
        B = 1
        src = src.view(B, -1) # (1, S)
        state = self.encoder(src, [S])
        inp = th.LongTensor([[self.vocab("<bos>")]]*B).view(1, 1).to('cpu') # (1, 1)
        inp = self.encoder.embedding(inp).view(B, -1)
        gen = ""
        for t in range(S):
            emb, out, state = self.decoder(inp, state)
            _, idx = out.max(dim=1, keepdim=True)
            # Normal argmax
            inp = self.encoder.embedding(idx.view(B, 1)).view(B, -1)
            # Use embeddinh version input
            # inp = emb.view(B, -1)
            gen += self.vocab.idx2word[idx.item()] + " "

        return gen
    
    

        

        
            


