from nltk import tokenize
from collections import Counter
class Vocab:
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
        self.count = 0

    def __call__(self, word):
       return self.word2idx[word] if word in self.word2idx else -1

    def add(self, word):
       if word in self.word2idx:
           return 

       self.word2idx[word] = self.count
       self.idx2word[self.count] = word
       self.count += 1

    def __len__(self):
        return len(self.word2idx)

    

        


