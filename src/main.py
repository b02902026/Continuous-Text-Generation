from AE import *
from util import *
from dataloader import *
from preprocess import *
import torch as th
import os
import pickle
from torch import nn, optim
import random
from vs_autoencoder import Autoencoder
from modules import *
from gan import GAN

def main():
    
    if not os.path.exists(os.path.join('../data/', 'vocab.pkl')):
        raise Exception('No data !')
    
    with open('../data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    trainset = NEWS(path='../data/train.json', vocab=vocab)
    valset = NEWS(path='../data/val.json', vocab=vocab)
    dataloader = get_dataloader(trainset, batch_size=20, shuffle=True)
    valloader = get_dataloader(valset, batch_size=5, shuffle=False)
    encoder = Encoder(emb_size=200, hidden_size=300, vocab_size=len(vocab))
    decoder = Decoder(emb_size=200, hidden_size=300, embedding_layer=encoder.embedding, vocab_size=len(vocab))
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab('<pad>'), reduction='sum')
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    VSAE = Autoencoder(vocab, 200)
    #optimizer = optim.Adam(VSAE.parameters(), lr=1e-4)
    autoencoder = SentenceAE(encoder, decoder, vocab, loss_fn, optimizer, dataloader, VSAE)
    
            
    for e in range(100):
        print("\n{}Epoch {}{}".format('-'*20, e, '-'*20))
        autoencoder.encoder.train()
        autoencoder.VSAE.train()
        autoencoder.train(dataloader)
        print("\nVALIDATION")
        autoencoder.encoder.eval()
        autoencoder.VSAE.eval()
        autoencoder.train(valloader, training=False)
        print("\nSanity check")
        for _ in range(6):
            v = trainset
            idx = random.randint(0, len(v)-1)
            gen = autoencoder.inference(th.LongTensor(v[idx]+[vocab("<eos>")]))
            print("original:", " ".join([vocab.idx2word[x] for x in v[idx]]))
            print("generated:", gen)
        ''' 
        if e % 10 == 0:
            th.save(autoencoder.encoder.state_dict(), "encoder_{}".format(e))
            th.save(autoencoder.decoder.state_dict(), "decoder_{}".format(e))
        '''

    # train GAN
    generator = autoencoder.decoder
    discriminator = Discriminator(300, [2,3,4,5])
    gan = GAN(generator, discriminator, autoencoder, vocab)
    for e in range(100):
        print("\n{}Epoch {}{}".format('-'*20, e, '-'*20))
        gan.train_generator(dataloader)
        if e % 5 == 0:
            gan.train_discriminator(dataloader)


if __name__ == "__main__":
    main()
