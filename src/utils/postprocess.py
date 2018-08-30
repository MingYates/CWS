# -*- coding: utf-8 -*-
import numpy as np


def parsetags(sent, tseq):
    tmp = ''
    binlist = tag2bin(tseq)
    for i, b in enumerate(binlist):
        if b == 0:
            tmp += '#|#'
            tmp += sent[i]
        else:
            tmp += sent[i]
    seg_w = tmp.split('#|#')
    seg_w = [s for s in seg_w if len(s) > 0]

    seg_b = []
    sidx = 0
    for s in seg_w:
        eidx = sidx + len(s)
        seg_b.append((sidx, eidx))
        sidx = eidx
    return seg_w, seg_b


def tag2bin(tseq):
    result = [int(t in ['I', 'E']) for t in tseq]
    # print(result)
    return result


def int2tag(pred, lens, t2i):
    result = []
    i2t = dict([(v, k) for k, v in t2i.items()])
    for p, l in zip(pred, lens):
        p = p[:l]
        result.append([i2t[t] for t in p])
    return result


def int2word(sents, lens, c2i):
    result = []
    i2c = dict([(v, k) for k, v in c2i.items()])
    for s, l in zip(sents, lens):
        s = s[:l]
        result.append([i2c[c] for c in s])
    return result


def bin2tag(bseq):
    result = []
    for i in range(len(bseq) - 1):
        cur = bseq[i]
        pst = bseq[i + 1]
        if cur == 0:
            if pst == 0:
                result.append('U')
            else:
                result.append('B')
        else:
            if pst == 0:
                result.append('E')
            else:
                result.append('I')

    last = bseq[-1]
    if last == 0:
        result.append('U')
    else:
        result.append('E')
    # print(result)
    return result


def tofile(sents, preds, labels, path):
    gfile = open(path + '.gold', 'w', encoding='utf8')
    pfile = open(path + '.pred', 'w', encoding='utf8')

    for s, p, l in zip(sents, preds, labels):
        # seg result
        gsegs, _ = parsetags(sent=s, tseq=l)
        gfile.write('  '.join(gsegs) + '\n')
        psegs, _ = parsetags(sent=s, tseq=p)
        pfile.write('  '.join(psegs) + '\n')

    gfile.close()
    pfile.close()


