import torch as th
from AE import SentenceAE
from seq2seq import Encoder, Decoder
import os

def training(data_dir):
    news_train = NEWS(data_path=os.path.join(data_dir, '.train.pkl'), vocab_path='../data/vocab.pkl')
    encoder, decoder = Encoder(emb_size=300, hidden_size=512, vocab_size=len(news_train.vocab)), Decoder()
    SAE = SentenceAE()
