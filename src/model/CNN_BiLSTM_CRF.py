# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import numpy as np
import pickle


class CWS_CBiCRF:
    def __init__(self, ntags, embed_mat, embed_dim):
        self.ntags = ntags
        self.embed_weights = embed_mat
        self.embed_dim = embed_dim
        self.build()

    def build(self):
        with tf.name_scope('inputs'):
            self.in_x = tf.placeholder(tf.int32, shape=[None, None], name='chars')
            self.in_y = tf.placeholder(tf.int32, shape=[None, None], name='tags')
            self.in_len = tf.placeholder(tf.int32, shape=None, name='lens')
            self.in_mxl = tf.placeholder(tf.int32, shape=None, name='batch_maxlen')
            self.dropout = tf.placeholder_with_default(0.0, [], name='dropout')
            self.lr = tf.placeholder_with_default(1e-3, [], name='learningrate')

        with tf.name_scope('embedding'):
            self.W_embed = tf.Variable(self.embed_weights, dtype=tf.float32, name='embed_weights', trainable=False)
            embed = tf.nn.embedding_lookup(self.W_embed, self.in_x, name='embedding')
            embed = tf.reshape(embed, shape=[-1, self.in_mxl, self.embed_dim])

        with tf.name_scope('cnn'):
            fsizes = [2, 3, 4]
            n_units = 64
            conv = []
            for i in fsizes:
                filter_i = tf.Variable(tf.random_normal([i, self.embed_dim, n_units]))
                conv_i = tf.nn.relu(tf.nn.conv1d(embed, filters=filter_i, stride=1, padding='SAME'))
                conv.append(conv_i)
            conv = tf.concat(conv, axis=-1)

        with tf.name_scope('bilstm'):
            _input = tf.concat([embed, conv], axis=-1)

            outdim_lstm = 256  # dim of lstm out
            self.cell_fw = tf.nn.rnn_cell.BasicLSTMCell(outdim_lstm//2, name='cell_fw')
            self.cell_bw = tf.nn.rnn_cell.BasicLSTMCell(outdim_lstm//2, name='cell_bw')
            _bilstm, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs=_input, dtype=tf.float32,
                                                         sequence_length=tf.reshape(self.in_len, shape=[-1]))
            _bilstm = tf.concat(_bilstm, axis=-1)
            _bilstm = tf.reshape(_bilstm, shape=[-1, outdim_lstm])

        with tf.name_scope('proj'):
            outdim_proj = 128
            lasthid = tf.layers.dense(inputs=_bilstm, units=outdim_proj, use_bias=True, activation=tf.nn.tanh,
                                      kernel_regularizer=tf.nn.l2_loss)
            lasthid = tf.nn.dropout(lasthid, keep_prob=1-self.dropout)

        with tf.name_scope('logits'):
            logits = tf.layers.dense(inputs=lasthid, units=self.ntags, use_bias=False, kernel_regularizer=tf.nn.l2_loss)
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
            self.summary = tf.summary.merge_all()

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def input(self, data, lr=1e-3, dropout=0.0):
        # bi, bmxlen, c_batch, t_batch, m_batch, l_batch
        bmxlen = data[1]
        bx = data[2]
        by = data[3]
        # bm = data[4]
        bl = data[5]
        # print(bm.shape)

        feed_dict = dict()
        feed_dict.update({
            'inputs/chars:0': bx,
            'inputs/tags:0': by,
            'inputs/batch_maxlen:0': bmxlen,
            'inputs/dropout:0': dropout,
            'inputs/learningrate:0': lr
        })
        feed_dict['inputs/lens:0'] = bl
        return feed_dict
