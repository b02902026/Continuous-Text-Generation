from nltk import tokenize
from collections import defaultdict, Counter
import pickle
import argparse
import os
from nltk.tokenize import TweetTokenizer
from util import *
import json

def run_preprocess(args):
    tokenized_sentences = []
    inverse_index = defaultdict(list)
    line_idx = 0
    counter = Counter()
    tokenizer = TweetTokenizer()
    with open(args.corpus_path) as f:
        for i, line in enumerate(f):
            tokens = tokenizer.tokenize(line.replace("\"",""))
            if len(tokens) < args.sent_threshold:
                continue
            tokenized_sentences.append(tokens)
            for word in tokens:
                inverse_index[word].append(line_idx)
                counter[word] += 1
            line_idx += 1
            
    # remove the low frequency word and sentences contain them / build vocabulary
    vocab = Vocab()
    vocab.add('<pad>')
    vocab.add('<bos>')
    vocab.add('<eos>')
    vocab.add('<unk>')
    for word, freq in counter.items():
        if freq < args.word_threshold:
            for sent_idx in inverse_index[word]:
                tokenized_sentences[sent_idx] = []
        else:
            vocab.add(word)
    # remove empty lines
    tokenized_sentences = [s for s in tokenized_sentences if s]
    train_split = int(len(tokenized_sentences) * 0.8)
    print(len(tokenized_sentences[train_split:]))
    # save the vocabulary and processed sentences
    with open(os.path.join(args.save_dir,"vocab.pkl"), 'wb') as f:
        pickle.dump(vocab, f)
    with open(os.path.join(args.save_dir,"train.json"), 'w') as f:
        json.dump({"sentences":tokenized_sentences[:train_split]}, f)
    with open(os.path.join(args.save_dir,"val.json"), 'w') as f:
        json.dump({"sentences":tokenized_sentences[train_split:]}, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus_path', default='../data/news.2007.en.shuffled')
    parser.add_argument('-save_dir', default='../data/')
    parser.add_argument('-word_threshold', type=int, default=10)
    parser.add_argument('-sent_threshold', type=int, default=5)
    args = parser.parse_args()
    run_preprocess(args)


