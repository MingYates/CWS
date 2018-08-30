# -*- coding: utf-8 -*-
import numpy as np


class MDICT:
    def __init__(self, wlist=None):
        if wlist is not None:
            self.construct(wlist)

    def load(self, _input, mode='file'):
        wlist = []
        if mode == 'list':
            wlist = _input
        elif mode == 'file':
            with open(_input, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    wlist.append(line.split(' ')[0])
        else:
            print('unknown data, input a list please.')
        self.construct(wlist)
        print('load dict done.')

    def construct(self, wlist):
        tmpdict = {}
        for w in wlist:
            c = w[0]
            if c not in tmpdict:
                tmpdict[c] = [w]
            else:
                tmpdict[c].append(w)
        # remove copies
        for c, ws in tmpdict.items():
            tmpdict[c] = set(tmpdict[c])
        self.dict = tmpdict

    def match(self, sent, maxlen=5):
        result = {'sent': sent, 'matches': []}
        for i in range(len(sent)):
            sdict = []
            if sent[i] in self.dict:
                sdict = self.dict[sent[i]]

            for j in range(maxlen):
                eidx = min(i+j, len(sent))
                seg = sent[i:eidx]
                if seg in sdict:
                    _m = {'w': seg, 'match': (i, eidx)}
                    result['matches'].append(_m)
        return result

    def encode_simple(self, sent, maxlen=5, mode='norm'):
        """
        :param sent: sentences
        :param maxlen: max length of words
        :param mode: bin {0,1} or norm [0-1]
        :param padlen:
        :return:
        """
        matchs = np.zeros(shape=(3, len(sent)), dtype=np.int32)
        for i in range(len(sent)):
            sdict = []
            if sent[i] in self.dict:
                sdict = self.dict[sent[i]]

            for j in range(1, maxlen):  # word length >= 2
                eidx = min(i+j, len(sent)-1)  # exactly the end index
                seg = sent[i:eidx+1]
                if seg in sdict:
                    matchs[0, i] += 1  # for B
                    matchs[1, i+1:eidx] += 1  # for I
                    matchs[2, eidx] += 1  # for E
        if mode == 'norm':
            matchs_exp = np.exp(matchs) + 1
            matchs_code = np.divide(matchs_exp, np.reshape(np.sum(matchs_exp, axis=-1), newshape=(-1, 1)))
        else:
            matchs_code = np.greater_equal(matchs, 1).astype(np.int32)
        return matchs_code

    def encode_full(self, sent, maxlen=5):
        matchs_e = np.zeros(shape=len(sent), dtype=np.int32)
        matchs_i = np.zeros(shape=(maxlen-1, len(sent)), dtype=np.int32)
        matchs_x = np.zeros(shape=(maxlen-1, len(sent)), dtype=np.int32)
        for i in range(len(sent)):
            sdict = []
            if sent[i] in self.dict:
                sdict = self.dict[sent[i]]

            for j in range(1, maxlen):
                eidx = min(i+j, len(sent)-1)
                seg = sent[i:eidx+1]
                if seg in sdict:
                    matchs_e[i] += 1  # for B
                    matchs_i[j-1, i+1:eidx+1] += 1  # for I
                    matchs_x[j-1, eidx] += 1  # for E
        matchs = np.vstack((matchs_e, matchs_i, matchs_x)).astype(np.int32)
        return matchs

    def encode_board(self, sent, maxlen=5, padlen=None):
        matchs = np.zeros(shape=((maxlen-1)*2, len(sent)), dtype=np.int32)
        for i in range(len(sent)):
            sdict = []
            if sent[i] in self.dict:
                sdict = self.dict[sent[i]]

            for j in range(1, maxlen):
                eidx = min(i+j, len(sent)-1)
                seg = sent[i:eidx+1]
                if seg in sdict:
                    matchs[(j-1)*2, i] = 1
                    matchs[(j-1)*2+1, eidx] = 1
        return matchs

    def encode(self, sent, maxlen=5, mode='simple'):
        if mode == 'full':
            return self.encode_full(sent=sent, maxlen=maxlen)
        elif mode == 'board':
            return self.encode_board(sent=sent, maxlen=maxlen)
        else:
            return self.encode_simple(sent=sent, maxlen=maxlen)

    def batch_encode(self, data, maxlen=5, mode='simple'):
        sents = [''.join(i[0]) for i in data]
        maxlen = max([len(s) for s in sents])

        batchms = []
        for s in sents:
            if mode == 'simple':
                ms = self.encode_simple(s, maxlen, mode='bin')
                # print(ms)
                if len(s) < maxlen:
                    pad = np.zeros(shape=(ms.shape[0], maxlen-len(s)))
                    ms = np.hstack((ms, pad)).astype(np.int32)
                    # print(ms)
                batchms.append(ms)
        return np.asarray(batchms)


if __name__ == '__main__':
    data = ['杭州', '西湖', '风景', '旅游', '旅游胜地']
    # data = ['杭州', '西湖', '风景', '旅游胜地']
    mdict = MDICT(wlist=data)

    sent = [list('杭州西湖风景很好，是旅游胜地！'),
            list('杭州是旅游胜地!')]

    # print(mdict.match(sent))
    # print(mdict.encode_simple(sent, mode='bin'))
    # print(mdict.encode_board(sent))
    # print(mdict.encode_full(sent))

    print(mdict.batch_encode(sent, mode='simple'))
    print('done.')

