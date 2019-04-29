from AE import *
from util import *
from dataloader import *
from preprocess import *
from seq2seq import *
import torch as th
import os
import pickle
from torch import nn, optim
import random

def main():
    
    if not os.path.exists(os.path.join('../data/', 'vocab.pkl')):
        raise Exception('No data !')
    
    with open('../data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    trainset = NEWS(path='../data/train.json', vocab=vocab)
    valset = NEWS(path='../data/val.json', vocab=vocab)
    print(len(valset))
    dataloader = get_dataloader(trainset, batch_size=20, shuffle=True)
    valloader = get_dataloader(valset, batch_size=25, shuffle=False)
    encoder = Encoder(emb_size=300, hidden_size=512, vocab_size=len(vocab))
    decoder = Decoder(emb_size=300, hidden_size=1024, vocab_size=len(vocab))
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab('<pad>'), reduction='sum')
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-4)
    autoencoder = SentenceAE(encoder, decoder, vocab, loss_fn, optimizer, dataloader)
    
    for e in range(100):
        print("{}Epoch {}{}".format('-'*20, e, '-'*20))
        autoencoder.train(dataloader)
        print("VALIDATION")
        autoencoder.train(valloader, trainable=False)
        print("\nSanity check")
        for _ in range(5):
            idx = random.randint(0, len(valset)-1)
            gen = autoencoder.inference(th.LongTensor(valset[idx]))
            print("original:", " ".join([vocab.idx2word[x] for x in valset[idx]]))
            print("generated:", gen)

        if e % 10 == 0:
            th.save(autoencoder.encoder.state_dict(), "encoder_{}".format(e))
            th.save(autoencoder.decoder.state_dict(), "decoder_{}".format(e))

if __name__ == "__main__":
    main()
