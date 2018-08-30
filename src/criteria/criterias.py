# -*- coding: utf-8 -*-
import numpy as np
from utils.postprocess import *


def div_safe(a, b):
    """
    safe div
    :param a: positive num
    :param b: positive num
    :return: div a/b
    """
    if a == 0:
        return 0.0
    elif b == 0:
        return a / (b + 1e-8)
    else:
        return a / b


def metrics(sents, preds, labels, t2i):
    total_pre = 0
    total_gth = 0
    total_crt = 0

    cmat = np.zeros(shape=(len(t2i), len(t2i)), dtype=np.int32)
    for s, p, l in zip(sents, preds, labels):
        for _i, _j in zip(l, p):
            cmat[t2i[_i], t2i[_j]] += 1
        # seg result
        wseg_p, segs_p = parsetags(sent=s, tseq=p)
        wseg_l, segs_l = parsetags(sent=s, tseq=l)
        # if not wseg_p == wseg_l:
        #     print('P:' + '|'.join(wseg_p))
        #     print('T:' + '|'.join(wseg_l))
        total_pre += len(segs_p)
        total_gth += len(segs_l)
        total_crt += len([1 for _i in segs_p if _i in segs_l])

    precision = div_safe(total_crt, total_pre)
    recall = div_safe(total_crt, total_gth)
    f1_score = div_safe(2 * precision * recall, (precision + recall))

    infos = 'precision={}, recall={}, f1={}\n'.format(precision, recall, f1_score)
    for t, i in t2i.items():
        ipre = div_safe(np.sum(cmat[i,i]), np.sum(cmat[:,i]))
        ircl = div_safe(np.sum(cmat[i,i]), np.sum(cmat[i,:]))
        if1 = div_safe(2*ipre*ircl, ipre + ircl)
        infos += '{}: p={}, r={}, f={}\n'.format(t, ipre, ircl, if1)
    return precision, recall, f1_score, infos


