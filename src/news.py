import os
import io
import json
import torch as th
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from utils import vocab
import picjle

class NEWS(Dataset):
    def __init__(self, data_path, vocab_path):
        super().__init__()
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        reruen len(self.data)
    
    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x), reverse=True)
        lengths = [len(x) for x in batch]
        max_len = lengths[0]
        PAD = self.vocab('<pad>')
        batch = [x + [PAD for _ in range(max_length - len(x))] for x in batch]
        batch_map = {}
        batch_map['sentence'] = th.LongTensor(batch)
        batch_map['lengths'] = th.LongTensor(lengths)
        
        return batch_map

