import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.distributions.normal import Normal

class GAN:
    def __init__(self, generator, discriminator, autoencoder, vocab):
        self.generator = generator
        self.discriminator = discriminator
        self.vocab = vocab
        self.autoencoder = autoencoder
        self.optimizer_G = Adam(generator.parameters(), lr=1e-3)
        self.optimizer_D = Adam(discriminator.parameters(), lr=1e-3)
    
    def to_categorical(self, x):
        B, S = x.size()
        onehot = th.zeros(x.size(0), S, len(self.vocab))
        return onehot.scatter_(2, x.unsqueeze(2), 1.).float()

    def mask(self, lengths, max_len):
        mask = []
        V = len(self.vocab)
        for b in lengths:
            mask.append(th.cat((th.ones(b, V), th.zeros(max_len-b, V)), 0))
        
        mask = th.stack(mask, 0)
        return mask

    def sample(self, B, inference=False):
        H = self.generator.hidden_size
        prior_h = Normal(th.zeros(B, H), th.ones(B, H)) # standard normal prior
        prior_c = Normal(th.zeros(B, H), th.ones(B, H)) # standard normal prior
        gen = []
        inp = th.LongTensor([self.vocab('<bos>')]*B).view(B, 1)
        inp = self.to_categorical(inp)
        state = (prior_h.sample().unsqueeze(0), prior_c.sample().unsqueeze(0))
        lengths = [0 for _ in range(B)]
        sentence = [[] for _ in range(B)]
        stopped = [0 for _ in range(B)]
        for t in range(40):
            prob, state = self.generator(inp, state)
            idx = prob.squeeze(1).max(dim=-1)[1]
            for i in range(B):
                if idx[i] == self.vocab("<eos>") or t == 39:
                    lengths[i] = t+1
                    stopped[i] = 1
                if inference and not stopped[i]:
                    sentence[i].append(self.vocab.idx2word[i])

            gen.append(prob.squeeze(1))
            inp = prob
        
        
        gen = th.stack(gen, 1) # (B, max_len, V)
        gen = gen * self.mask(lengths, 40)
        if inference:
            return sentence

        return gen, th.LongTensor(lengths)

    def train_generator(self, dataloader, training = True):
        H = self.generator.hidden_size
        total_loss = 0
        for b, (real, lengths) in enumerate(dataloader):
            self.optimizer_G.zero_grad()
            gen, gen_lengths = self.sample(real.size(0)) 
            loss = -th.sum(self.discriminator(gen, gen_lengths)) # (B, 1)
            if training:
                loss.backward()
                self.optimizer_G.step()
             
            total_loss += loss.item()
            print("\rGenerator:batch:{}/{}, loss:{}, total loss:{}".format(b, len(dataloader), loss.item(), total_loss / (b + 1)), end='')

    def train_discriminator(self, dataloader, training=True):
        H = self.generator.hidden_size
        total_loss = 0
        for b, (real, lengths) in enumerate(dataloader):
            self.optimizer_D.zero_grad()
            gen, gen_lengths = self.sample(real.size(0))
            fake = self.discriminator(gen, gen_lengths)
            real = self.discriminator(self.to_categorical(real), lengths)
            loss =  -th.log(real) - th.log(1-fake)   # (B, 1)
            loss = th.sum(loss)
            if training:
                loss.backward()
                self.optimizer_D.step()

            total_loss += loss.item()
            print("\rDiscriminator:batch:{}/{}, loss:{}, total loss:{}".format(b, len(dataloader), loss.item(), total_loss / (b + 1)), end='')
    
    def inference(self, d = 300):
        B = 20
        z = th.distributions.normal.Normal(th.zeros(d), th.ones(d))
        inp = th.LongTensor([vocab('<bos>')]*B).view(B, 1)
        max_len = 20
        state = (th.randn(1, B, self.generator.hidden_size), th.randn(1, B, self.generator.hidden_size))
        output_embedding = []
        for t in range(max_len):
            out, state = self.generator(z, inp, state)
            output_embedding.append(out)
            inp = out

