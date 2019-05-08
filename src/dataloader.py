import torch as th
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from util import Vocab

class NEWS(Dataset):
    def __init__(self, path, vocab):
        with open(path, 'r') as f:
            self.data = json.load(f)["sentences"]
        
        self.vocab = vocab
    
    def __getitem__(self, idx):
        s = [self.vocab(x) for x in self.data[idx]]
        return s
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        batch.sort(key=lambda x: len(x), reverse=True)
        lengths = [len(x) + 1 for x in batch]
        max_len = lengths[0]
        PAD = self.vocab('<pad>')
        EOS = self.vocab('<eos>')
        s = [x + [EOS] + [PAD]*(max_len - len(x)) for x in batch]
        
        return th.LongTensor(s), th.LongTensor(lengths)

def get_dataloader(dataset, batch_size, shuffle):

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)



        

