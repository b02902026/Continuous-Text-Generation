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
    

    def mask(self, lengths):
        max_len = max(lengths)
        mask = []
        V = len(self.vocab)
        for b in lengths:
            mask.append(th.cat((th.ones(b, V), th.zeros(max_len-b, V)), 0))

        return th.stack(mask, 0)

    def sample(self, src):
        B, S = src.size()
        H = self.generator.hidden_size
        prior_h = Normal(th.zeros(B, H), th.ones(B, H)) # standard normal prior
        prior_c = Normal(th.zeros(B, H), th.ones(B, H)) # standard normal prior
        gen = []
        inp = th.LongTensor([self.vocab('<bos>')]*B).view(B, 1)
        inp = self.autoencoder.encoder.embedding(inp).view(B, -1)
        state = (prior_h.sample(), prior_c.sample())
        lengths = [0 for _ in range(B)]
        for t in range(40):
            prob, state = self.generator(inp, state)
            idx = prob.max(dim=1)[1]
            for i in idx:
                if i == self.vocab("<eos>"):
                    lengths[i] = t

            gen.append(prob)
            inp = prob.view(B, -1)
        
        gen = th.stack(gen, 0) # (B, max_len, V)
        gen = gen * self.mask(lengths)

        return gen

    def train_generator(self, dataloader, training = True):
        H = self.generator.hidden_size
        for b, (real, lengths) in enumerate(dataloader):
            self.optimizer_G.zero_grad()
            gen = self.sample(real) 
            loss = -self.discriminator(gen) # (B, 1)
            if training:
                loss.backward()
                self.optimizer_G.step()

    def train_discriminator(self, dataloader=True):
        H = self.generator.hidden_size
        for b, (real, lengths) in enumerate(dataloader):
            self.optimizer_D.zero_grad()
            gen = self.sample(real)
            fake = self.discriminator(gen)
            real = self.discriminator(real)
            loss =  -th.log(real) - th.log(1-fake)   # (B, 1)
            if training:
                loss.backward()
                self.optimizer_D.step()

    
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

