import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam

class GAN:
    def __init__(self, generator, discriminator, vocab):
        self.generator = generator
        self.discriminator = discriminator
        self.vocab = vocab

    def training(self, dataloader):


    def inference(self, d = 300):
        z = th.distributions.normal.Normal(th.zeros(d), th.ones(d))
        inp = th.LongTensor([vocab('<bos>')]*B).view(1, 1)
        max_len = 20
        for t in range(max_len):
            self.generator(z)
        

