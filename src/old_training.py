import torch as th
from data import *
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from dataloader import *
from utils import *
from copy import deepcopy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from seq2seq import *

def train_seq2seq(train, dev, input_vocab, output_vocab, args):
    trainloader = get_dataloader(train, args.batch_size)
    devloader = get_dataloader(dev, args.batch_size)
    epoch = args.epochs
    loss_fn = nn.CrossEntropyLoss(ignore_index = 0, reduction = "sum")
    encoder = Encoder(hidden_size = args.hidden_size, vocab_size = len(input_vocab), emb_size = args.input_dim)
    decoder = Decoder(hidden_size = args.hidden_size * 2, vocab_size = len(output_vocab), emb_size = args.output_dim, attn=args.style)
    optim = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = 1e-3)
    for e in range(epoch):
        total_loss = 0
        for i, (src, target, length) in enumerate(trainloader):
            B = src.size(0)
            max_seq_len = target.size(1)
            optim.zero_grad()
            out, h = encoder(src, length)
            loss = 0
            inp = th.LongTensor([output_vocab.index_of("<SOS>") for _ in range(B)]).view(B, 1)
            for t in range(max_seq_len):
                if args.att:
                    pred, h = decoder(inp, h, out, length)
                else:
                    pred, h = decoder(inp, h)

                loss += loss_fn(pred, target[:,t])
                inp = target[:, t].unsqueeze(1)

            loss = loss / B
            loss.backward()
            optim.step()
            total_loss += loss.item()

            print("\repoch:{}/{}, batch:{}/{}, total loss: {}".format(e, epoch, i, len(trainloader), total_loss / (i + 1)), end='')
         
        val_loss = 0
        encoder.eval()
        decoder.eval()
        for i, (vsrc, vtgt, vl) in enumerate(devloader):
            B = vsrc.size(0)
            max_seq_len = vtgt.size(1)
            out, h = encoder(vsrc, vl)
            loss = 0
            inp = th.LongTensor([output_vocab.index_of("<SOS>") for _ in range(B)]).view(B, 1)
            for t in range(max_seq_len):
                if args.att:
                    pred, h = decoder(inp, h, out, vl)
                else:
                    pred, h = decoder(inp, h)

                loss += loss_fn(pred, vtgt[:,t])
                inp = pred.max(dim = 1)[1].view(B,1)

 
            loss = loss / B
            val_loss += loss.item()
        encoder.train()
        decoder.train()
        
        print()
        print("======VALIDATION======")
        print("epoch:{}/{}, total loss: {}".format(e, epoch, val_loss / len(devloader)))
        print("======================")

    return encoder, decoder

    

