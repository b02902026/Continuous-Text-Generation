from AE import *
from util import *
from dataloader import *
from preprocess import *
from seq2seq import *
import torch as th
import os
import pickle
from torch import nn, optim

def main():
    
    if not os.path.exists(os.path.join('../data/', 'vocab.pkl')):
        raise Exception('No data !')
    
    with open('../data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    dataset = NEWS(path='../data/corpus.pkl', vocab=vocab)
    dataloader = get_dataloader(dataset, batch_size=2, shuffle=True)
    encoder = Encoder(emb_size=200, hidden_size=512, vocab_size=len(vocab))
    decoder = Decoder(emb_size=200, hidden_size=1024, vocab_size=len(vocab))
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab('<pad>'), reduction='none')
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    autoencoder = SentenceAE(encoder, decoder, vocab, loss_fn, optimizer, dataloader)
    
    for e in range(100):
        print("{}Epoch {}{}".format('-'*20, e, '-'*20))
        autoencoder.train(dataloader)

    for _ in range(5):
        print(autoencoder.inference())

if __name__ == "__main__":
    main()
