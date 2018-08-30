# -*- coding: utf-8 -*-
import logging


def getlogger(path, name='CWS-LOG'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fhandler = logging.FileHandler(path)
    fhandler.setLevel(logging.INFO)

    shandler = logging.StreamHandler()
    shandler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s --  %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)

    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    return logger