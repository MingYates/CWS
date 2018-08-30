from utils.prepare import *
import tensorflow as tf
from utils.postprocess import int2tag, int2word, tofile
from criteria.criterias import metrics
from utils.utils import getlogger


def train(model, mname, params, traindata, c2i, t2i, valdata=None, logger=None, tbpath=None):
    BATCHSIZE = params['batchsize']
    EPOCH = params['epoch']
    DROPOUT = params['dropout']
    LEARNINGRATE = params['lr']
    CORPUSNAME = params['dataname']

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=10)  # train saver
        tbpath += '{}_{}_lr{}_dp{}_'.format(CORPUSNAME, mname, LEARNINGRATE, DROPOUT)
        trainwriter = tf.summary.FileWriter(tbpath+'train', graph=tf.get_default_graph())  # tensorboard log writer
        testwriter = tf.summary.FileWriter(tbpath+'test')
        sess.run(tf.global_variables_initializer())  # init ops
        sess.run(tf.local_variables_initializer())

        ## TRAINING
        MINLR = params['minlr']
        EARLYSTOP = params['earlystop']
        PATIENCE = params['patience']
        PSTEP = 0
        BESTLOSS = 1e10
        VBESTLOSS = 1e10
        VBESTf1 = 0.0
        for _epoch in range(EPOCH):
            eploss_list = []
            for data in gen_batch(traindata, c2i=c2i, t2i=t2i, batchsize=BATCHSIZE, shuffle=True):
                _bnum = data[0]
                _fdict = model.input(data, lr=LEARNINGRATE, dropout=DROPOUT)
                _, _iloss, _isummary = sess.run([model.train_op, model.loss, model.summary], feed_dict=_fdict)
                eploss_list.append(_iloss)  # batch loss
                trainwriter.add_summary(_isummary, global_step=_epoch * int(len(traindata) / BATCHSIZE) + _bnum)
                if _bnum % 10 == 0:
                    info = 'epoch={}, batch={}, loss={}'.format(_epoch, _bnum, _iloss)
                    logger.info(info)
                    print(info)
            # epoch info
            epoch_loss = np.sum(np.asarray(eploss_list))
            info = 'EPOCH {} DONE. LR={}, DROPOUT={}, LOSS={}'.format(_epoch, params['lr'], DROPOUT, epoch_loss)
            logger.info(info)
            print(info)

            # validation
            if valdata is not None:
                print('------------------------------------------')
                vresult = []
                valloss = 0.0
                for data in gen_batch(valdata, c2i=c2i, t2i=t2i, batchsize=BATCHSIZE, shuffle=False):
                    _bnum = data[0]
                    _fdict = model.input(data, dropout=0.0)
                    _ipred, _ivloss, _isummary = sess.run([model.pred, model.loss, model.summary], feed_dict=_fdict)
                    valloss += _ivloss
                    vresult.extend(_ipred.tolist())
                    testwriter.add_summary(_isummary, global_step=(_epoch+1) * int(len(valdata) / BATCHSIZE) + _bnum)
                    info = 'epoch={}, batch={}, val_loss={}'.format(_epoch, data[0], _ivloss)
                    logger.info(info)
                    print(info)

                if valloss < VBESTLOSS:
                    VBESTLOSS = valloss

                vlen = [d[-1] for d in valdata]
                vpred = int2tag(vresult, lens=vlen, t2i=t2i)
                sents = int2word([d[0].tolist() for d in valdata], lens=vlen, c2i=c2i)
                labels = int2tag([d[2].tolist() for d in valdata], lens=vlen, t2i=t2i)
                # evaluation
                _, _, f1, info = metrics(sents=sents, labels=labels, preds=vpred, t2i=t2i)
                logger.info(info)
                print(info)
                # save results
                oname = '{},model_{},dp_{},epoch_{},f1_{}'.format(CORPUSNAME, mname,
                                                                  round(DROPOUT, 2), _epoch, round(f1, 4))
                path = '/home/faust/PROJECTS/NEUTAG/data/predict/' + oname
                tofile(sents=sents, preds=vpred, labels=labels, path=path)

            # learning rate decay & early stop
            if epoch_loss < BESTLOSS:
                BESTLOSS = epoch_loss
                PSTEP = 0
            else:
                PSTEP += 1
                if PSTEP > PATIENCE:
                    if LEARNINGRATE > MINLR:
                        LEARNINGRATE *= 0.5
                        info = 'learning rate decay: lr={}'.format(LEARNINGRATE)
                        logger.info(info)
                        print(info)
                        PSTEP = 0
                    else:
                        if PSTEP > EARLYSTOP:
                            break


def main_all(TRAIN, TEST, params):
    # -------- log --------
    logpath = '/home/faust/PROJECTS/NEUTAG/CWS.log'
    logger = getlogger(logpath)

    # -------- tags --------
    tags = ['U', 'B', 'I', 'E']
    t2i = dict([(t, i) for i, t in enumerate(tags)])
    print(t2i)

    # -------- dict --------
    # dict match params
    dictpath = '/home/faust/PROJECTS/NEUTAG/data/dict'
    match_len = 5
    # match_mode = 'board'
    # match_dim = (match_len-1) * 2
    # # match_mode = 'full'
    # # match_dim = (match_len - 1) * 2 + 1
    # # match_mode = 'simple'
    # # match_dim = 3

    # -------- embedding --------
    dumppath = '/home/faust/PROJECTS/NEUTAG/data/char2vec.pkl'
    c2i, embed_mat, embed_dim = pickle.load(open(dumppath, "rb"))

    # -------- train --------
    for _mode, _dim in zip(['board', 'simple', 'full'], [(match_len - 1) * 2, 3, (match_len - 1) * 2 + 1]):
        TRAINDATA = prepare(TRAIN, c2i=c2i, t2i=t2i, dictpath=dictpath, match_len=match_len, match_mode=_mode)
        TESTDATA = prepare(TEST, c2i=c2i, t2i=t2i, dictpath=dictpath, match_len=match_len, match_mode=_mode)

        tf.reset_default_graph()
        from model.CWS_DR import CWS_DR
        info = '/////////////////////-*- CWS_DR -*-/////////////////////'
        logger.info(info)
        print(info)
        tbpath = '/files/faust/tf-data/NEUTAG/{}_'.format(_mode)
        model = CWS_DR(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim, match_dim=_dim)
        train(model, mname='CWS_DR', params=params,
              traindata=TRAINDATA, c2i=c2i, t2i=t2i, valdata=TESTDATA, logger=logger, tbpath=tbpath)

        # CWS_DR_ATT
        tf.reset_default_graph()
        from model.CWS_DR_ATT import CWS_DR
        info = '/////////////////////-*- CWS_DR_ATT -*-/////////////////////'
        logger.info(info)
        print(info)
        tbpath = '/files/faust/tf-data/NEUTAG/{}_'.format(_mode)
        model = CWS_DR(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim, match_dim=_dim)
        train(model, mname='CWS_DR_ATT', params=params,
              traindata=TRAINDATA, c2i=c2i, t2i=t2i, valdata=TESTDATA, logger=logger, tbpath=tbpath)

    # BiLSTM_CRF
    TRAINDATA = prepare(TRAIN, c2i=c2i, t2i=t2i, dictpath=dictpath, match_len=match_len)
    TESTDATA = prepare(TEST, c2i=c2i, t2i=t2i, dictpath=dictpath, match_len=match_len)

    tf.reset_default_graph()
    from model.BiLSTM_CRF import CWS_BiCRF
    info = '/////////////////////-*- BiLSTM_CRF -*-/////////////////////'
    logger.info(info)
    print(info)
    tbpath = '/files/faust/tf-data/NEUTAG/'
    model = CWS_BiCRF(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim)
    train(model, mname='BiLSTM_CRF', params=params,
          traindata=TRAINDATA, c2i=c2i, t2i=t2i, valdata=TESTDATA, logger=logger, tbpath=tbpath)


def main_single(TRAIN, TEST, params, mname='BiLSTM_CRF'):
    # -------- log --------
    logpath = '/home/faust/PROJECTS/NEUTAG/CWS.log'
    logger = getlogger(logpath)

    # -------- tags --------
    tags = ['U', 'B', 'I', 'E']
    t2i = dict([(t, i) for i, t in enumerate(tags)])
    print(t2i)

    # -------- dict --------
    # dict match params
    dictpath = '/home/faust/PROJECTS/NEUTAG/data/dict'
    match_len = 5
    # _mode = 'board'
    # _dim = (match_len-1) * 2
    _mode = 'full'
    _dim = (match_len - 1) * 2 + 1
    # _mode = 'simple'
    # _dim = 3

    # -------- embedding --------
    dumppath = '/home/faust/PROJECTS/NEUTAG/data/gram2vec.pkl'
    c2i, embed_mat, embed_dim = pickle.load(open(dumppath, "rb"))

    # -------- traindata --------
    TRAINDATA = prepare(TRAIN, c2i=c2i, t2i=t2i, dictpath=dictpath, match_len=match_len, match_mode=_mode)
    TESTDATA = prepare(TEST, c2i=c2i, t2i=t2i, dictpath=dictpath, match_len=match_len, match_mode=_mode)

    # info = '/////////////////////-*- CWS_DR_ATT -*-/////////////////////'
    # logger.info(info)
    # print(info)
    #
    # tf.reset_default_graph()
    # from model.CWS_DR_ATT import CWS_DR
    # tbpath = '/files/faust/tf-data/NEUTAG/{}_'.format(_mode)
    # model = CWS_DR(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim, match_dim=_dim)
    # train(model, mname='CWS_DR_ATT', params=params,
    #       traindata=TRAINDATA, c2i=c2i, t2i=t2i, valdata=TESTDATA, logger=logger, tbpath=tbpath)

    info = '/////////////////////-*- CWS_DC_ATT -*-/////////////////////'
    logger.info(info)
    print(info)

    tf.reset_default_graph()
    from model.CWS_DC_ATT import CWS_DC
    tbpath = '/files/faust/tf-data/NEUTAG/{}_'.format(_mode)
    model = CWS_DC(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim, match_dim=_dim)
    train(model, mname='CWS_DC_ATT', params=params,
          traindata=TRAINDATA, c2i=c2i, t2i=t2i, valdata=TESTDATA, logger=logger, tbpath=tbpath)

    # info = '/////////////////////-*- CWS_DR -*-/////////////////////'
    # logger.info(info)
    # print(info)
    #
    # tf.reset_default_graph()
    # from model.CWS_DR import CWS_DR
    # tbpath = '/files/faust/tf-data/NEUTAG/{}_'.format(_mode)
    # model = CWS_DR(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim, match_dim=_dim)
    # train(model, mname='CWS_DR', params=params,
    #       traindata=TRAINDATA, c2i=c2i, t2i=t2i, valdata=TESTDATA, logger=logger, tbpath=tbpath)

    # info = '/////////////////////-*- CWS_CBiCRF -*-/////////////////////'
    # logger.info(info)
    # print(info)
    #
    # tf.reset_default_graph()
    # from model.CNN_BiLSTM_CRF import CWS_CBiCRF
    # tbpath = '/files/faust/tf-data/NEUTAG/{}_'.format(_mode)
    # model = CWS_CBiCRF(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim)
    # train(model, mname='CWS_CBiCRF', params=params,
    #       traindata=TRAINDATA, c2i=c2i, t2i=t2i, valdata=TESTDATA, logger=logger, tbpath=tbpath)

    # info = '/////////////////////-*- BiLSTM_CRF -*-/////////////////////'
    # logger.info(info)
    # print(info)
    #
    # tf.reset_default_graph()
    # from model.BiLSTM_CRF import CWS_BiCRF
    # tbpath = '/files/faust/tf-data/NEUTAG/'
    # model = CWS_BiCRF(ntags=len(t2i), embed_mat=embed_mat, embed_dim=embed_dim)
    # train(model, mname='BiLSTM_CRF', params=params,
    #       traindata=TRAINDATA, c2i=c2i, t2i=t2i, valdata=TESTDATA, logger=logger, tbpath=tbpath)


if __name__ == '__main__':
    # -------- data --------
    path = '/home/faust/PROJECTS/NEUTAG/data/PKU/pku_data.pkl'
    # path = '/home/faust/PROJECTS/NEUTAG/data/AS/as_data.pkl'
    # path = '/home/faust/PROJECTS/NEUTAG/data/MSR/msr_data.pkl'
    # path = '/home/faust/PROJECTS/NEUTAG/data/CITYU/cityu_data.pkl'
    TRAIN, TEST = pickle.load(open(path, 'rb'))
    # TRAIN = TRAIN[:5000]
    # TEST = TEST[:200]
    print(len(TRAIN))
    print(len(TEST))

    # # prepare data
    # TRAINDATA = prepare(TRAIN, c2i=c2i, t2i=t2i, dictpath=dictpath, match_len=match_len, match_mode=match_mode)
    # TESTDATA = prepare(TEST, c2i=c2i, t2i=t2i, dictpath=dictpath, match_len=match_len, match_mode=match_mode)
    # TRAINDATA = TRAINDATA[:5000]
    # TESTDATA = TESTDATA[:200]
    # print(len(TRAINDATA))
    # print(len(TESTDATA))

    # model_name = 'CWS_DC_ATT'
    # params = {
    #     'dataname': 'pku',
    #     'batchsize': 128,
    #     'epoch': 10,
    #     'dropout': _dp,
    #     'lr': 0.002,
    #     'earlystop': 5,
    #     'patience': 1,  # patience for lr decay
    #     'minlr': 1e-8  # min learning rate
    # }
    #
    # # main_all(TRAIN=TRAIN, TEST=TEST, params=params)
    # main_single(TRAIN=TRAIN, TEST=TEST, params=params, mname=model_name)

    model_name = 'CWS_DC_ATT'
    for _dp in np.arange(0.0, 0.5, step=0.05):
        # -------- params --------
        params = {
            'dataname': 'pku',
            'batchsize': 128,
            'epoch': 10,
            'dropout': _dp,
            'lr': 0.002,
            'earlystop': 5,
            'patience': 1,  # patience for lr decay
            'minlr': 1e-8   # min learning rate
        }

        # main_all(TRAIN=TRAIN, TEST=TEST, params=params)
        main_single(TRAIN=TRAIN, TEST=TEST, params=params, mname=model_name)

