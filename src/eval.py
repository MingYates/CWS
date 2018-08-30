from utils.prepare import *
import tensorflow as tf
from model.CWS_DC import CWS_ATT


def evaluate(model, params, traindata, c2i, t2i, valdata=None):
    BATCHSIZE = params['batchsize']
    EPOCH = params['epoch']
    DROPOUT = params['dropout']
    LEARNINGRATE = params['lr']

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNINGRATE)
    train_op = optimizer.minimize(model.loss)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(tbpath, graph=tf.get_default_graph())  # tensorboard log
        # init
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # epoch
        for _epoch in range(EPOCH):
            for data in gen_batch(traindata, c2i=c2i, t2i=t2i, batchsize=BATCHSIZE, shuffle=True, cliplen=64):
                _bnum = data[0]
                _fdict = input(data, dropout=0.2, islen=True, ismask=False)
                _, _iloss, _isummary = sess.run([train_op, model.loss, model.summary], feed_dict=_fdict)

                writer.add_summary(_isummary, global_step=_epoch * int(len(traindata) / BATCHSIZE) + _bnum)
                if _bnum % 10 == 0:
                    print('epoch={}, batch={}, loss={}'.format(_epoch, _bnum, _iloss))

            if valdata is not None:
                valresult = []
                for k, _x, _y, _l in gen_batch(valdata, BATCHSIZE, model.maxlen):
                    fdict = model.input(_x, _y, _l)
                    ival, _ = sess.run([model.pred, model.summary], feed_dict=fdict)
                    valresult.append(ival.tolist())


if __name__ == '__main__':
    tags = ['B', 'I', 'E', 'U', '#']
    t2i = dict([(t, i) for i, t in enumerate(tags)])
    print(t2i)
    dumppath = '/home/faust/PROJECTS/NEUTAG/data/char2vec.pkl'
    c2i, embed_mat, embed_dim = pickle.load(open(dumppath, "rb"))

    # path = '/home/faust/PROJECTS/NEUTAG/data/PKU/pku_data.pkl'
    # path = '/home/faust/PROJECTS/NEUTAG/data/AS/as_data.pkl'
    path = '/home/faust/PROJECTS/NEUTAG/data/MSR/msr_data.pkl'
    # path = '/home/faust/PROJECTS/NEUTAG/data/CITYU/cityu_data.pkl'

    TRAIN_DATA, TEST_DATA = pickle.load(open(path, 'rb'))
    # print('max length in training set: %d' % max([len(ws) for ws, ts in TRAIN_DATA]))
    # print('max length in testing set: %d' % max([len(ws) for ws, ts in TEST_DATA]))
    maxlen = 64
    tbpath = '/files/faust/tf-data/NEUTAG'


    params = {
        'batchsize': 128,
        'epoch': 100,
        'dropout': 0.8,
        'lr': 0.01
    }
    model = CWS_ATT(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim)
    evaluate(model, params=params, traindata=TRAIN_DATA, c2i=c2i, t2i=t2i, valdata=TEST_DATA)

