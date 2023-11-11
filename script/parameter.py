#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  : dx
# @File    : parameter.py
# @Software: PyCharm
# @Note    :


class Para():
    def __init__(self, lr, rec, drop, batch_size, epoch, dev_ratio, test_ratio, embedding_dim, alpha1, alpha2, alpha3):
        self.lr = lr
        self.rec = rec
        self.drop = drop
        self.batch_size = batch_size
        self.epoch = epoch
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.embedding_dim = embedding_dim
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

