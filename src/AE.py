import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
class SentenceAE:
    def __init__(self, Encoder, Decoder, vocab, loss_fn, optimizer, dataloader):
        self.encoder = Encoder
        self.decoder = Decoder
        self.vocab = vocab
        self.loss_fn = loss_fn
        self.optim  = optimizer

    def train(self, dataloader, trainable=True, device='cpu'):
        total_loss = 0
        for i, (x, lengths) in enumerate(dataloader):
            loss = 0
            B, S = x.size()
            state = self.encoder(x, lengths) # (B, 2H)
            inp = th.LongTensor([[self.vocab("<bos>")]]*B).view(B, 1).to(device) # (B, 1)
            inp = self.encoder.embedding(inp).squeeze() # (B, E)
            for t in range(S):
                _ , logit, state = self.decoder(inp, state)
                loss += (self.loss_fn(logit, x[:,t]).sum() / lengths.sum().item())
                inp = self.encoder.embedding(x[:,t]).squeeze().detach()
                
            
            loss /= B
            if trainable:
                loss.backward()
                self.optim.step()

            total_loss += loss.item()
            if i % 20 == 0:
                print("batch:{}/{}, loss:{}, total loss:{}".format(i, len(dataloader), loss.item(), total_loss / (i + 1)))
            
    
    def inference(self, src):
        
        S = src.size(0)
        src = src.unqueeze(0) # (1, S)
        state = self.encoder(src, [S])
        inp = th.LongTensor([[vocab("<bos>")]]*B).view(1, 1).to(device) # (1, 1)
        gen = ""
        for t in range(S):
            emb, out, state = self.decoder(inp, state)
            _, idx = out.max(dim=1, keepdim=True)
            inp = self.encoder.embedding(idx).view(1, -1)
            gen += self.vocab.idx2word[idx.item()]

        return gen
    
    

        

        
            


