from collections import Counter

class Vocab(object):
    def __init__(self, sents):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        words = [w for s in sents for w in s]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.items():
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk
        self.word2idx = {}
        self.idx2word = []
        for w in v:
            self.word2idx[w] = len(self.word2idx)
            self.idx2word.append(w)

        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']
