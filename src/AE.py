import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from vs_autoencoder import Autoencoder

class SentenceAE:
    def __init__(self, Encoder, Decoder, vocab, loss_fn, optimizer, dataloader ,VSAE):
        self.encoder = Encoder
        self.decoder = Decoder
        self.vocab = vocab
        self.loss_fn = loss_fn
        self.optimizer  = optimizer
        self.VSAE = VSAE

    def train_vs(self, dataloader, training=True):

        total_loss = 0
        for i, (src, lengths) in enumerate(dataloader):
            self.optimizer.zero_grad()
            B, S = src.size()
            loss, _  = self.VSAE(src, lengths, self.loss_fn)
            loss /= lengths.sum().item()
            if training:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            print("\rbatch:{}/{}, loss:{}, total loss:{}".format(i, len(dataloader), loss.item(), total_loss / (i + 1)), end='')
    

    def get_mask(self, lengths):
        S = lengths[0].item()
        mask = th.zeros(lengths.size(0), S)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        return mask.float()
    
    def get_noise(self, sampler, B):
        noise = []
        for i in range(B):
            noise.append(sampler .sample())

        return th.stack(noise, 0)

    def train(self, dataloader, training=True, device='cpu'):

        total_loss = 0
        aug_loss_fn = nn.CosineEmbeddingLoss(reduction='none') 
        alpha = 0.6
        sampler = th.distributions.gumbel.Gumbel(loc=th.zeros(len(self.vocab)), scale=th.ones(len(self.vocab)))
        for i, (x, lengths) in enumerate(dataloader):
            loss = 0
            aug_loss = []
            #mask = self.get_mask(lengths)
            B, S = x.size()
            state = self.encoder(x, lengths) # (B, 2H)
            inp = th.LongTensor([[self.vocab("<bos>")]]*B).view(B, 1).to(device) # (B, 1)
            #inp = th.zeros(B, len(self.vocab)).scatter_(1, inp, 1) #+ self.get_noise(sampler, B)
            #noise = self.get_noise(sampler, B)
            for t in range(S):
                prob, state = self.decoder(inp, state)
                loss += self.loss_fn(prob, x[:,t])
                # Normal seq2seq (gumbel)
                #inp = th.zeros(B, len(self.vocab)).scatter_(1, x[:, t].view(B, 1), 1) #+ noise
                inp = x[:, t].view(B, 1)
                
                
                # Training using embedding
                #inp = vec.view(B, -1)

            loss /= lengths.sum().item()
            if training:
                loss.backward()
                self.optimizer.step()
            
            
            total_loss += loss.item()
            print("\rbatch:{}/{}, loss:{}, total loss:{}".format(i, len(dataloader), loss.item(), total_loss / (i + 1)), end='')
            
    
    def inference(self, src):
        
        S = src.size(0)
        B = 1
        src = src.view(B, -1) # (1, S)
        state = self.encoder(src, [S])
        inp = th.LongTensor([[self.vocab("<bos>")]]*B).view(1, 1).to('cpu') # (1, 1)
        #inp = th.zeros(B, len(self.vocab)).scatter_(1, inp, 1) #+ self.get_noise(sampler, B)
        gen = ""
        for t in range(50):
            out, state = self.decoder(inp, state)
            _, idx = out.max(dim=1, keepdim=True)
            # Normal argmax
            inp = idx
            # Use embeddinh version input
            #inp = emb.view(B, -1)
            if self.vocab.idx2word[idx.item()] == "<eos>":
                break
            gen += self.vocab.idx2word[idx.item()] + " "

        return gen
    
    

        

        
            


