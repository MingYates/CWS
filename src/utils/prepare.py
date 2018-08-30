# -*- coding: utf-8 -*-
import pickle
import numpy as np
import random
from dict.MDICT import MDICT
import math


# def trans2int_pad(data, c2i, t2i, cliplen=None):
#     """
#     transform words and tags into indexs
#     :param data:
#     :param c2i:
#     :param t2i:
#     :param cliplen:
#     :return:
#     """
#     words = []
#     tags = []
#     lens = []
#
#     bmxl = -1
#     if cliplen is None:
#         bmxl = max([len(w) for w, _ in data])
#     else:
#         bmxl = min(max([len(w) for w, _ in data]), cliplen)
#
#     for cs, ts in data:
#         cs = cs[:bmxl]
#         ics = [c2i['NONE']] * bmxl
#         ics[:len(cs)] = map(lambda c: c2i[c] if c in c2i else c2i['OOV'], cs)
#
#         ts = ts[:bmxl]
#         its = [t2i['U']] * bmxl
#         its[:len(ts)] = [t2i[t] for t in ts]
#         if len(ts) == len(cs):
#             words.append(ics)
#             tags.append(its)
#             lens.append(len(cs))
#     words = np.reshape(np.asarray(words), newshape=(-1, bmxl))
#     tags = np.reshape(np.asarray(tags), newshape=(-1, bmxl))
#     lens = np.asarray(lens)
#     return words, tags, lens, bmxl


# def trans2int(data, c2i, t2i, cliplen=None):
#     """
#     transform words and tags into indexs
#     :param data:
#     :param c2i:
#     :param t2i:
#     :param cliplen:
#     :return:
#     """
#     # 加载词典
#     path = '/home/faust/PROJECTS/NEUTAG/data/dict'
#     mdict = MDICT()
#     mdict.load(path, mode='file')
#
#     result = []
#     c2ifun = lambda c: c2i[c] if c in c2i else c2i['OOV']
#     for cs, ts in data:
#         ics = [c2ifun(c) for c in cs]
#         its = [t2i[t] for t in ts]
#         isent = ''.join(cs)
#         ims = mdict.encode_simple(isent, maxlen=5, mode='bin')  # dict match features
#         if len(ts) == len(cs):
#             result.append((np.asarray(ics), np.asarray(its), ims, len(cs)))
#     return result


# # 加载词典
# path = '/home/faust/PROJECTS/NEUTAG/data/dict'
# mdict = MDICT()
# mdict.load(path, mode='file')


def prepare(data, c2i, t2i, dictpath, match_len=5, match_mode='simple'):
    # 加载词典
    mdict = MDICT()
    mdict.load(dictpath, mode='file')

    result = []
    uni2ifun = lambda c: c2i[c] if c in c2i else c2i['OOV1']
    bi2ifun = lambda c: c2i[c] if c in c2i else c2i['OOV2']
    for cs, ts in data:
        unis = [uni2ifun(c) for c in cs]
        bis = [bi2ifun(_i + _j) for _i, _j in zip(cs[:-1], cs[1:])]
        its = [t2i[t] for t in ts]
        isent = ''.join(cs)
        ms = mdict.encode(isent, maxlen=match_len, mode=match_mode)  # dict match features
        if len(ts) == len(cs) > 0:
            result.append((np.asarray(unis), np.asarray(bis), np.asarray(its), ms.astype(np.int32), len(cs)))
    return result


def padding(seq, pvalue=0, plen=64):
    result = seq
    if len(seq) < plen:
        if len(seq.shape) > 1:
            _pad = np.full(shape=(seq.shape[0], plen - seq.shape[1]), fill_value=pvalue)
            result = np.hstack((result, _pad)).astype(np.int32)
        else:
            _pad = np.full(shape=(plen - seq.shape[0]), fill_value=pvalue)
            result = np.hstack((result, _pad)).astype(np.int32)
    return result


def gen_batch(data, c2i, t2i, batchsize=64, shuffle=True):
    if shuffle:
            random.shuffle(data)

    bnum = math.ceil(len(data)/batchsize)
    for bi in range(int(bnum)):
        idx_s = bi*batchsize
        idx_e = min(len(data), (bi+1)*batchsize)
        batch = data[idx_s:idx_e]
        if len(batch) > 0:
            bmxlen = max([i[-1] for i in batch])
            ci_batch = []
            bi_batch = []
            t_batch = []
            m_batch = []
            l_batch = []
            for uis, bis, its, ims, ilen in batch:
                ci_batch.append(padding(uis, pvalue=c2i['PAD'], plen=bmxlen))
                bi_batch.append(padding(bis, pvalue=c2i['PAD'], plen=bmxlen))
                t_batch.append(padding(its, pvalue=t2i['U'], plen=bmxlen))
                m_batch.append(padding(ims, pvalue=0, plen=bmxlen))
                l_batch.append(ilen)
            ci_batch = np.reshape(np.asarray(ci_batch), newshape=(len(batch), bmxlen))
            bi_batch = np.reshape(np.asarray(bi_batch), newshape=(len(batch), bmxlen))
            t_batch = np.reshape(np.asarray(t_batch), newshape=(len(batch), bmxlen))
            m_batch = np.stack(m_batch, axis=0).astype(np.int32)
            l_batch = np.reshape(np.asarray(l_batch), newshape=(len(batch)))
            # print(ci_batch.shape)
            # print(bi_batch.shape)
            # print(t_batch.shape)
            # print(m_batch.shape)
            # print(l_batch.shape)
            yield bi, bmxlen, ci_batch, bi_batch, t_batch, m_batch, l_batch


# def gen_batch(data, c2i, t2i, batchsize=64, shuffle=True, cliplen=None):
#     """
#     generate batch for input
#     :param data:
#     :param c2i:
#     :param t2i:
#     :param batchsize:
#     :param shuffle:
#     :param cliplen:
#     :return:
#     """
#     if shuffle:
#         random.shuffle(data)
#
#     for bi in range(int(len(data)/batchsize)):
#         idx_s = bi*batchsize
#         idx_e = min(len(data), (bi+1)*batchsize)
#         batch = data[idx_s:idx_e]
#         xs, ts, ls, mxl = trans2int_pad(batch, c2i=c2i, t2i=t2i, cliplen=cliplen)
#         ms = mdict.batch_encode(data, mode='simple')  # dict match features
#         yield (bi, xs, ts, ls, ms, mxl)  # batch index, char index, tag index, len, dict match features, batch maxlen
#

# def input(data, lr=1e-3, dropout=0.0, islen=True, ismask=False):
#     """
#     prepare input for model, return feeddict
#     :param data:
#     :param dropout:
#     :param islen:
#     :param ismask:
#     :return:
#     """
#     _, bx, by, bl, bmxl = data
#     feed_dict = {
#         'inputs/chars:0': bx,
#         'inputs/tags:0': by,
#         'inputs/batch_maxlen:0': bmxl,
#         'inputs/dropout:0': dropout,
#         'inputs/learningrate:0': lr
#     }
#     if islen:
#         feed_dict['inputs/lens:0'] = bl
#
#     if ismask:
#         mask = []
#         for ilen in bl:
#             imask = [0]*bmxl
#             imask[:ilen] = [1]*ilen
#             mask.append(ismask)
#         mask = np.reshape(np.asarray(mask), newshape=(-1, bmxl))
#         feed_dict['inputs/inmask:0'] = mask
#     return feed_dict

