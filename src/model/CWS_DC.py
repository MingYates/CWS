# -*- coding: utf-8 -*-
import tensorflow as tf
import random
import numpy as np
import pickle


class CWS_DC:
    def __init__(self, ntags, embed_mat, embed_dim, match_dim):
        self.ntags = ntags
        self.embed_weights = embed_mat
        self.embed_dim = embed_dim
        self.match_dim = match_dim
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
            embed = tf.nn.dropout(embed, keep_prob=1-self.dropout)

        with tf.name_scope('bilstm'):
            outdim_lstm = 200  # dim of lstm out
            self.cell_fw = tf.nn.rnn_cell.BasicLSTMCell(outdim_lstm//2, name='cell_fw')
            self.cell_bw = tf.nn.rnn_cell.BasicLSTMCell(outdim_lstm//2, name='cell_bw')
            _bilstm, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs=embed, dtype=tf.float32,
                                                         sequence_length=tf.reshape(self.in_len, shape=[-1]))
            _bilstm = tf.concat(_bilstm, axis=-1)
            _bilstm = tf.reshape(_bilstm, shape=[-1, outdim_lstm])
            _bilstm = tf.nn.dropout(_bilstm, keep_prob=1-self.dropout)

        with tf.name_scope('proj'):
            outdim_proj = 128
            lasthid = tf.layers.dense(inputs=_bilstm, units=outdim_proj, use_bias=True, activation=tf.nn.tanh)
            lasthid = tf.nn.dropout(lasthid, keep_prob=1-self.dropout)

        with tf.name_scope('logits'):
            logits = tf.layers.dense(inputs=lasthid, units=self.ntags, use_bias=False)
            logits = tf.reshape(logits, shape=[-1, self.in_mxl, self.ntags], name='logits')

        with tf.name_scope('attention'):
            with tf.name_scope('cnn'):
                self.in_m = tf.cast(tf.reshape(self.in_m, shape=[-1, self.in_mxl, self.match_dim]), dtype=tf.float32)
                fsizes = [1, 2, 3, 4]
                ksizes = [4, 4, 4, 4]
                conv_1 = []
                for i, j in zip(fsizes, ksizes):
                    filter_i = tf.Variable(tf.random_normal([i, self.match_dim, j]))
                    mconv_i = tf.nn.conv1d(self.in_m, filters=filter_i, stride=1, padding='SAME')
                    conv_1.append(mconv_i)
                conv_1 = tf.nn.sigmoid(tf.concat(conv_1, axis=-1, name='conv_concat'))
            conv_1 = tf.reshape(conv_1, shape=[-1, sum(ksizes)])
            attw = tf.layers.dense(conv_1, self.ntags, activation=tf.nn.sigmoid)
            attw = tf.reshape(attw, shape=[-1, self.in_mxl, self.ntags])
            attw = tf.nn.dropout(attw, keep_prob=1-self.dropout)
            logits = tf.multiply(attw, logits, name='att_logits')

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


    # def build(self):
    #     # embedding weights
    #     self.W_embed = tf.Variable(self.embed_weights, dtype=tf.float32, trainable=False, name='embedding_weights')
    #     with tf.name_scope('inputs'):
    #         self.in_x = tf.placeholder(tf.int32, shape=[None, None, 1], name='chars')
    #         self.in_y = tf.placeholder(tf.int32, shape=[None, None], name='tags')
    #         self.in_len = tf.placeholder(tf.int32, shape=[None], name='lens')
    #
    #     with tf.name_scope('embedding'):
    #         embed = tf.nn.embedding_lookup(self.W_embed, self.in_x, name='embedding')
    #
    #     with tf.name_scope('conv_and_pool'):
    #         conv_1 = []
    #         for i in [1, 2, 3, 4, 5]:
    #             filter_i = tf.Variable(tf.random_normal([i, self.embed_dim, 32]))
    #             conv_1_i = tf.nn.conv1d(tf.reshape(embed, shape=[-1, self.in_mxl, self.embed_dim]),
    #                                     filters=filter_i, stride=1, padding='SAME')
    #             conv_1_i = tf.nn.relu(conv_1_i)
    #             conv_1.append(conv_1_i)
    #         conv_1 = tf.concat(conv_1, axis=-1, name='conv_concat')
    #         pool_1 = tf.reduce_max(conv_1, axis=1, name='pool_max')
    #
    #     with tf.name_scope('global_attention'):
    #         self.W_att_0 = tf.Variable(tf.random_normal([32*5, 1]))
    #         self.W_att_1 = tf.Variable(tf.random_normal([32*5, 1]))
    #         wet_0 = tf.matmul(tf.reshape(conv_1, shape=[-1, 32*5]), self.W_att_0)
    #         wet_1 = tf.matmul(tf.reshape(pool_1, shape=[-1, 32*5]), self.W_att_1)
    #         wet = tf.reshape(tf.add(tf.reshape(wet_0, shape=[-1, self.in_mxl, 1]),
    #                                 tf.reshape(wet_1, shape=[-1, 1, 1])),
    #                          shape=[-1, self.in_mxl, 1])
    #         weighted = tf.multiply(wet, conv_1)
    #
    #     with tf.name_scope('concat'):
    #         self.feat = tf.concat([conv_1, weighted], axis=-1)
    #
    #     with tf.name_scope('last_dense'):
    #         self.W_last = tf.Variable(tf.random_normal([32*5*2, 64]))
    #         self.b_last = tf.Variable(tf.random_normal([1, 64]))
    #         wx = tf.matmul(tf.reshape(self.feat, shape=[-1, 32*5*2]), self.W_last)
    #         wx_b = tf.nn.relu(tf.add(wx, self.b_last))
    #         lasthidden = tf.reshape(wx_b, shape=[-1, self.in_mxl, 64])
    #
    #     with tf.name_scope('gen_props'):
    #         self.W_prop = tf.Variable(tf.random_normal([64, self.ntags]))
    #         self.b_prop = tf.Variable(tf.random_normal([1, self.ntags]))
    #         wx = tf.matmul(tf.reshape(lasthidden, shape=[-1, 64]), self.W_prop)
    #         wx_b = tf.nn.softmax(tf.add(wx, self.b_prop), axis=-1)
    #         self.prop = tf.reshape(wx_b, shape=[-1, self.in_mxl, self.ntags])
    #
    #     with tf.name_scope('crf'):
    #         self.log_likelihood, self.transmat = tf.contrib.crf.crf_log_likelihood(self.prop, self.in_y, self.in_len)
    #         viterbi_seqs, _ = tf.contrib.crf.crf_decode(self.prop, self.transmat, self.in_len)
    #         self.pred = tf.reshape(viterbi_seqs, [-1, self.in_mxl], name='ner_predict')
    #
    #     with tf.name_scope('loss'):
    #         self.loss = tf.reduce_mean(-self.log_likelihood, name='loss')
    #         tf.summary.scalar('loss', self.loss)
    #
    #     with tf.name_scope('summary'):
    #         self.summary = tf.summary.merge_all()

# if __name__ == '__main__':
#     tags = ['B', 'I', 'E', 'U', '#']
#     t2i = dict([(t, i) for i, t in enumerate(tags)])
#     print(t2i)
#     dumppath = '/home/faust/PROJECTS/NEUTAG/data/char2vec.pkl'
#     c2i, emat, edim = pickle.load(open(dumppath, "rb"))
#
#     # path = '/home/faust/PROJECTS/NEUTAG/data/PKU/pku_data.pkl'
#     # path = '/home/faust/PROJECTS/NEUTAG/data/AS/as_data.pkl'
#     path = '/home/faust/PROJECTS/NEUTAG/data/MSR/msr_data.pkl'
#     # path = '/home/faust/PROJECTS/NEUTAG/data/CITYU/cityu_data.pkl'
#
#     TRAIN_DATA, TEST_DATA = pickle.load(open(path, 'rb'))
#     # print('max length in training set: %d' % max([len(ws) for ws, ts in TRAIN_DATA]))
#     # print('max length in testing set: %d' % max([len(ws) for ws, ts in TEST_DATA]))
#     maxlen = 64
#     TRAIN_DATA = trans2int(TRAIN_DATA, c2i=c2i, t2i=t2i, maxlen=maxlen)
#     TEST_DATA = trans2int(TEST_DATA, c2i=c2i, t2i=t2i, maxlen=maxlen)
#
#     tbpath = '/files/faust/tf-data/NEUTAG'
#     model = CWS_ATT(maxlen=maxlen, char2int=c2i, tag2int=t2i, char2vec=emat, edim=edim, tbpath=tbpath)
#
#     BATCHSIZE = 128
#     EPOCH = 50
#     DROPOUT = 0.5
#     model.train(TRAIN_DATA, batchsize=BATCHSIZE, epoch=EPOCH, dropout=DROPOUT)

