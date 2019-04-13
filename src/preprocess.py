from nltk import tokenize
from collections import defaultdict, Counter
import pickle
import argparse
import os

def run_preprocess(args):
    tokenized_sentences = []
    inverse_index = defaultdict(list)
    line_idx = 0
    counter = Counter()
    with open(args.corpus_path) as f:
        for line in f:
            tokens = line.strip().split()
            tokens = tokenize(tokens)
            if len(tokens) < args.sent_threshold:
                continue
            tokenized_sentence.append(tokens)
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
    tokenized_sentences = [s if s for s in tokenized_sentences]
    
    # save the vocabulary and processed sentences
    with open(os.path.join(args.save_dir,"vocab.pkl"), 'wb') as f:
        pickle.dump(f, vocab)
    with open(os.path.join(args.save_dir,"corpus.pkl"), 'wb') as f:
        pickle.dump(f, tokenized_sentences)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus_path', default='../data/news.2007.en.shuffled')
    parser.add_arguemnt('-save_dir', default='../data/')
    parser.add_argument('-word_threshold', type=int, default=4050)
    parser.add_argument('-sent_threshold', type=int, default=20)
    args = parser.parse_arg()
    run_preprocess(args)


