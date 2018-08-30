# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import numpy as np
import pickle


class CWS_GCNN:
    def __init__(self, ntags, embed_mat, embed_dim):
        self.ntags = ntags
        self.embed_weights = embed_mat
        self.embed_dim = embed_dim
        self.build()

    def build(self):
        with tf.name_scope('inputs'):
            self.in_x = tf.placeholder(tf.int32, shape=[None, None], name='chars')
            self.in_y = tf.placeholder(tf.int32, shape=[None, None], name='tags')
            self.in_m = tf.placeholder(tf.int32, shape=[None, None, None], name='dictmatch')
            self.in_len = tf.placeholder(tf.int32, shape=None, name='lens')
            self.in_mxl = tf.placeholder(tf.int32, shape=None, name='batch_maxlen')
            self.dropout = tf.placeholder_with_default(0.0, [], name='dropout')
            self.lr = tf.placeholder_with_default(1e-3, [], name='learningrate')

        with tf.name_scope('embedding'):
            self.W_embed = tf.Variable(self.embed_weights, dtype=tf.float32, name='embed_weights', trainable=False)
            embed = tf.nn.embedding_lookup(self.W_embed, self.in_x, name='embedding')
            embed = tf.reshape(embed, shape=[-1, self.in_mxl, self.embed_dim])

        with tf.name_scope('gated_cnn'):
            gconv, gcndim = self.gcnn(embed, indim=self.embed_dim, wsize=[1, 2, 3, 5], outdim=64, layers=3)
            gconv = tf.nn.dropout(gconv, keep_prob=1-self.dropout)

        with tf.name_scope('proj'):
            indim_proj = gcndim
            outdim_proj = 128
            lasthid = tf.layers.dense(inputs=tf.reshape(gconv, shape=[-1, indim_proj]),
                                      units=outdim_proj, use_bias=True, activation=tf.nn.tanh)
            lasthid = tf.nn.dropout(lasthid, keep_prob=1-self.dropout)

        with tf.name_scope('logits'):
            logits = tf.layers.dense(inputs=lasthid, units=self.ntags, use_bias=False)
            logits = tf.reshape(logits, shape=[-1, self.in_mxl, self.ntags], name='logits')

        with tf.name_scope('crf'):
            self.log_likelihood, self.transmat = tf.contrib.crf.crf_log_likelihood(logits, self.in_y, self.in_len)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-self.log_likelihood, name='loss')

        with tf.name_scope('pred'):
            viterbi_seqs, _ = tf.contrib.crf.crf_decode(logits, self.transmat, self.in_len)
            self.pred = tf.reshape(viterbi_seqs, [-1, self.in_mxl], name='ner_predict')

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('pred', self.pred)
            self.summary = tf.summary.merge_all()

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def gcnn(self, _input, indim, wsize, outdim=64, layers=5):
        # layer 1
        with tf.name_scope('gcnn_conv_1'):
            oconv_0 = []
            gconv_0 = []
            for _ws in wsize:
                ofilter_0 = tf.Variable(tf.random_normal([_ws, indim, outdim]), name='filter_oconv_0')
                oconv_0.append(tf.nn.conv1d(_input, filters=ofilter_0, stride=1, padding='SAME'))
                gfilter_0 = tf.Variable(tf.random_normal([_ws, indim, outdim]), name='filter_gconv_0')
                gconv_0.append(tf.nn.sigmoid(tf.nn.conv1d(_input, filters=gfilter_0, stride=1, padding='SAME')))
            conv_0 = tf.multiply(tf.concat(oconv_0, axis=-1), tf.concat(gconv_0, axis=-1))

        if layers < 2:
            return conv_0, outdim * len(wsize)
        else:
            result = [conv_0]
            _input = conv_0
            _indim = outdim * len(wsize)
            for i in range(1, layers):
                with tf.name_scope('gcnn_conv_1'):
                    oconv_i = []
                    gconv_i = []
                    for _ws in wsize:
                        ofilter_i = tf.Variable(tf.random_normal([_ws, _indim, outdim]), name='filter_oconv_' + str(i))
                        oconv_i.append(tf.nn.conv1d(_input, filters=ofilter_i, stride=1, padding='SAME'))
                        gfilter_i = tf.Variable(tf.random_normal([_ws, _indim, outdim]), name='filter_gconv_' + str(i))
                        gconv_i.append(tf.nn.sigmoid(tf.nn.conv1d(_input, filters=gfilter_i, stride=1, padding='SAME')))
                    conv_i = tf.multiply(tf.concat(oconv_i, axis=-1), tf.concat(gconv_i, axis=-1))
                    result.append(conv_i)
                    _input = conv_i
            return tf.concat(result, axis=-1), outdim * len(wsize) * layers

    def input(self, data, lr=1e-3, dropout=0.0):
        # bi, bmxlen, c_batch, t_batch, m_batch, l_batch
        bmxlen = data[1]
        bx = data[2]
        by = data[3]
        bm = data[4]
        bl = data[5]
        # print(bm.shape)

        feed_dict = dict()
        feed_dict.update({
            'inputs/chars:0': bx,
            'inputs/tags:0': by,
            'inputs/dictmatch:0': bm,
            'inputs/batch_maxlen:0': bmxlen,
            'inputs/dropout:0': dropout,
            'inputs/learningrate:0': lr
        })
        feed_dict['inputs/lens:0'] = bl
        return feed_dict

