import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import nn
from torch.autograd import Variable
import numpy as np
import nltk
import pickle

class Autoencoder(nn.Module):
    def __init__(self, vocab, embedding_size, dropout_p=0.5, bidirectional=True, is_vae=False):
        super(Autoencoder,self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = 200
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.vocab = vocab

        self.is_vae = is_vae
        self.encoder = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, dropout=dropout_p, bidirectional=bidirectional)
        self.decoder = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, dropout=dropout_p)
        self.mean = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.var  = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self, templates, lengths, loss_fn):
        """
        parameters:
            inp: [batch_size, seq_len]
        """
        B = templates.size(0)
        seq_max_len = torch.max(lengths)

        mu, logvar = self.encode(templates)
        z = self.reparameterize(mu, logvar)
        z = z.view(1, B, self.hidden_size)
        if self.is_vae:
            output = self.decode(z, templates, lengths)
        else:
            logvar = logvar.view(1, B, self.hidden_size)
            output = self.decode(logvar, templates, lengths)

        loss = loss_fn(output, templates.view(-1))
        kld = torch.mean(torch.sum(0.5 * (mu**2 + torch.exp(logvar) - logvar -1))).squeeze()
        
        return loss, (kld if self.is_vae else kld.zero_())

    def encode(self, templates):
        embedded = self.embedding(templates)
        output, hidden = self.encoder(embedded)
        hidden = hidden.transpose(0, 1).contiguous().view(-1, self.hidden_size*2)
        #hidden = hidden.contiguous().view(-1, self.hidden_size)
        if self.is_vae:
            mu = self.mean(hidden)
            logvar = self.var(hidden)
            return mu, logvar
        else:
            mu = self.mean(hidden)
            hidden = self.var(hidden)
            return mu, hidden

    def reparameterize(self, mu, logvar, training=True):
        if training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, hidden, target, lengths):
        SOS = torch.LongTensor([self.vocab('<bos>')]*target.size(0)).view(target.size(0), 1)
        inp = torch.cat((SOS, target[:, :-1]), -1)
        embedded = self.embedding(inp)
        output, hidden = self.decoder(embedded, hidden)
        output = self.fc(output)
        output = output.view(-1, self.vocab_size)
        return output



    def inference(self, templates):
        max_seq_len = templates.size(0)
        B = 1
        templates = templates.view(B, -1)
        mu, logvar = self.encode(templates)
        z = self.reparameterize(mu, logvar, training=False)
        if self.is_vae:
            hidden = z.view(1, 5, self.hidden_size)
        else:
            hidden = logvar.view(1, B, self.hidden_size)

        SOS= torch.LongTensor([1]*B).view(B, 1)
        input_id = SOS
        inp = self.embedding(SOS)
        words = []
        for t in range(max_seq_len):
            inp = inp.view(B, 1, self.embedding_size)
            output, hidden = self.decoder(inp, hidden)
            output = self.fc(output)
            output = F.softmax(output,-1)
            # eliminate <unk>
            prob, idx = output.view(B, -1).topk(2)
            for k in range(B):
                if idx[k,0] == self.vocab('<unk>'):
                    idx[k,0] = idx[k,1]
            words.append(idx[:,0])
            inp = Variable(idx[:,0]).view(B, 1).long()
            inp = self.embedding(inp)

        words = torch.stack(words, 1)
        story = []
        for sent in words:
            storyline = [self.vocab.idx2word[w.item()]+" " if w != self.vocab('<unk>') else "UNK " for w in sent]
            try:
                stop_index = storyline.index('<end> ')
                storyline = storyline[:stop_index]
            except:
                #print('no eos')
                pass
            storyline.append("\n")
            story.extend(storyline[:])

        return "".join(story)[:-1]

    def get_latent_feature(self, templates):
        B = templates.size(0)
        seq_max_len = templates.size(-1)
        templates = templates.view(B*5, seq_max_len)

        _, hidden = self.encode(templates)
        return hidden
