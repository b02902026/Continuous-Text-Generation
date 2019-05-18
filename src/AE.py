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

    def to_categorical(self, x):
        B, S = x.size()
        onehot = th.zeros(x.size(0), S, len(self.vocab))
        return onehot.scatter_(2, x.unsqueeze(2), 1.).float()

    def train(self, dataloader, training=True, device='cpu'):

        total_loss = 0
        for i, (x, lengths) in enumerate(dataloader):
            B, S = x.size()
            #sampler = th.distributions.gumbel.Gumbel(th.zeros(S, len(self.vocab)), th.ones(S, len(self.vocab)))
            loss = 0
            aug_loss = []
            onehot = self.to_categorical(x)
            state = self.encoder(onehot, lengths) # (B, 2H)
            inp = th.LongTensor([[self.vocab("<bos>")]]*B).view(B, 1).to(device) # (B, 1)
            inp = th.cat((inp, x[:,:-1]), 1)

            inp = self.to_categorical(inp)
            prob, _ = self.decoder(inp, (state[0].unsqueeze(0), state[1].unsqueeze(0)))
            prob = F.log_softmax(prob, -1)
            loss = self.loss_fn(prob.view(B*S,-1),x.view(B*S))
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
        onehot = self.to_categorical(src)
        state = self.encoder(onehot, [S])
        inp = th.LongTensor([[self.vocab("<bos>")]]*B).view(1, 1).to('cpu') # (1, 1)
        inp = self.to_categorical(inp)
        gen = ""
        state = (state[0].unsqueeze(0), state[1].unsqueeze(0))
        for t in range(50):
            out, state = self.decoder(inp, state)
            _, idx = out.squeeze(1).max(dim=1, keepdim=True)
            # Normal argmax
            inp = F.softmax(out, -1)
            #inp = self.to_categorical(idx)
            # Use embeddinh version input
            #inp = emb.view(B, -1)
            gen += self.vocab.idx2word[idx.item()] + " "
            if self.vocab.idx2word[idx.item()] == "<eos>":
                break

        return gen
    
    

        

        
            


