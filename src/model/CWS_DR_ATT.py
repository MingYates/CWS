# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import numpy as np
import pickle


class CWS_DR:
    def __init__(self, ntags, embed_mat, embed_dim, match_dim):
        self.ntags = ntags
        self.embed_weights = embed_mat
        self.embed_dim = embed_dim
        self.match_dim = match_dim
        self.build()

    def build(self):
        with tf.name_scope('inputs'):
            self.in_ch = tf.placeholder(tf.int32, shape=[None, None], name='chars')
            self.in_bi = tf.placeholder(tf.int32, shape=[None, None], name='bigram')
            self.in_y = tf.placeholder(tf.int32, shape=[None, None], name='tags')
            self.in_m = tf.placeholder(tf.int32, shape=[None, None, None], name='dictmatch')
            self.in_len = tf.placeholder(tf.int32, shape=None, name='lens')
            self.in_mxl = tf.placeholder(tf.int32, shape=None, name='batch_maxlen')
            self.dropout = tf.placeholder_with_default(0.0, [], name='dropout')
            self.lr = tf.placeholder_with_default(1e-3, [], name='learningrate')

        with tf.name_scope('embedding'):
            self.W_embed = tf.Variable(self.embed_weights, dtype=tf.float32, name='embed_weights', trainable=True)
            with tf.name_scope('char'):
                embed_c = tf.nn.embedding_lookup(self.W_embed, self.in_ch, name='embedding')
                embed_c = tf.reshape(embed_c, shape=[-1, self.in_mxl, self.embed_dim])
            with tf.name_scope('bigram'):
                embed_b = tf.nn.embedding_lookup(self.W_embed, self.in_bi, name='embedding')
                embed_b = tf.reshape(embed_b, shape=[-1, self.in_mxl, self.embed_dim])
            embed = tf.concat([embed_c, embed_b], axis=-1)

        with tf.name_scope('bilstm_c'):
            outdim_lstm = 256  # dim of lstm out
            self.cell_fw = tf.nn.rnn_cell.BasicLSTMCell(outdim_lstm//2, name='cell_fw')
            self.cell_bw = tf.nn.rnn_cell.BasicLSTMCell(outdim_lstm//2, name='cell_bw')
            _bilstm_c, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs=embed, dtype=tf.float32,
                                                           sequence_length=tf.reshape(self.in_len, shape=[-1]))
            _bilstm_c = tf.concat(_bilstm_c, axis=-1)
            _bilstm_c = tf.reshape(_bilstm_c, shape=[-1, outdim_lstm])

        with tf.name_scope('proj'):
            outdim_proj = 128
            lasthid = tf.layers.dense(inputs=_bilstm_c, units=outdim_proj, use_bias=True, activation=tf.nn.tanh,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.005))
            lasthid = tf.nn.dropout(lasthid, keep_prob=1-self.dropout)

        with tf.name_scope('logits'):
            logits = tf.layers.dense(inputs=lasthid, units=self.ntags, use_bias=True, activation=tf.nn.relu,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.005))
            logits = tf.reshape(logits, shape=[-1, self.in_mxl, self.ntags], name='logits')

        with tf.name_scope('attention'):
            with tf.name_scope('bilstm-m'):
                self.in_m = tf.cast(tf.reshape(self.in_m, shape=[-1, self.in_mxl, self.match_dim]), dtype=tf.float32)

                outdim_lstm_att = 16  # dim of lstm out
                self.cell_fw_att = tf.nn.rnn_cell.BasicLSTMCell(outdim_lstm_att//2, name='cell_fw_att')
                self.cell_bw_att = tf.nn.rnn_cell.BasicLSTMCell(outdim_lstm_att//2, name='cell_bw_att')
                _bilstm_m, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw_att, self.cell_bw_att,
                                                             inputs=self.in_m, dtype=tf.float32,
                                                             sequence_length=tf.reshape(self.in_len, shape=[-1]))
                _bilstm_m = tf.concat(_bilstm_m, axis=-1)
                _bilstm_m = tf.reshape(_bilstm_m, shape=[-1, outdim_lstm_att])

            att_in = tf.concat([_bilstm_m, lasthid], axis=-1)
            att_out = tf.layers.dense(att_in, self.ntags, activation=tf.nn.sigmoid,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.005))
            att_out = tf.reshape(att_out, shape=[-1, self.in_mxl, self.ntags])
            logits = tf.multiply(att_out, logits, name='att_logits')

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
        bc = data[2]
        bb = data[3]
        by = data[4]
        bm = data[5]
        bl = data[6]
        # print(bm.shape)

        feed_dict = dict()
        feed_dict.update({
            'inputs/chars:0': bc,
            'inputs/bigram:0': bb,
            'inputs/tags:0': by,
            'inputs/dictmatch:0': bm,
            'inputs/batch_maxlen:0': bmxlen,
            'inputs/dropout:0': dropout,
            'inputs/learningrate:0': lr
        })
        feed_dict['inputs/lens:0'] = bl
        return feed_dict

