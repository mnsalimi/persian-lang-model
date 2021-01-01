import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'sample.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'sample.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'sample.test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids