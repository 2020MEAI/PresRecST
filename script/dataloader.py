#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  : dx
# @File    : dataloader.py
# @Software: PyCharm
# @Note    :
import pandas as pd
from script.tools import *
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def process_dataset(dev_ratio, test_ratio, batch_size):
    # load data
    tcm_lung = pd.read_excel('data/TCM-Lung.xlsx')
    data_amount = tcm_lung.shape[0]

    # data_proccess
    nwfe_list = [x for x in range(data_amount)]
    x_train, x_test = train_test_split(nwfe_list, test_size=(dev_ratio+test_ratio), shuffle=False, random_state=2022)
    print("train_size:", len(x_train), "test_size:", len(x_test))

    sym_list = [[0]*1801 for _ in range(data_amount)]
    sym_array = np.array(sym_list)
    syn_list = [[0]*54 for _ in range(data_amount)]
    syn_array = np.array(syn_list)
    herb_list = [[0]*410 for _ in range(data_amount)]
    herb_array = np.array(herb_list)
    treat_list = [[0]*67 for _ in range(data_amount)]
    treat_array = np.array(treat_list)

    for i in range(data_amount):
        # sym
        sym_indexes = list(map(int, tcm_lung.iloc[i, 0].split(',')))
        sym_indexes = [x for x in sym_indexes if x < 1801]
        sym_array[i, sym_indexes] = 1
        # syn
        syn_indexes = list(map(int, tcm_lung.iloc[i, 1].split(',')))
        syn_indexes = [x for x in syn_indexes if x < 54]
        syn_array[i, syn_indexes] = 1
        # treat
        treat_indexes = list(map(int, tcm_lung.iloc[i, 2].split(',')))
        treat_indexes = [x for x in treat_indexes if x < 67]
        treat_array[i, treat_indexes] = 1
        # herb
        herb_indexes = list(map(int, tcm_lung.iloc[i, 3].split(',')))
        herb_indexes = [x for x in herb_indexes if x < 410]
        herb_array[i, herb_indexes] = 1

    train_dataset = PreDatasetLung4(sym_array[x_train], syn_array[x_train], treat_array[x_train], herb_array[x_train])
    test_dataset = PreDatasetLung4(sym_array[x_test], syn_array[x_test], treat_array[x_test], herb_array[x_test])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return x_train, x_test, train_loader, test_loader
