#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  : dx
# @File    : dataloader.py
# @Software: PyCharm
# @Note    :
import pandas as pd
import tqdm
from script.tools import *
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def process_dataset_tcmpd(dev_ratio, test_ratio, batch_size):
    # load data
    tcm_lung = pd.read_csv('data/prescript_1195.csv')
    data_amount = tcm_lung.shape[0]

    # data_proccess
    nwfe_list = [x for x in range(data_amount)]
    x_train, x_test = train_test_split(nwfe_list, test_size=(dev_ratio+test_ratio), shuffle=False, random_state=2022)
    print("train_size:", len(x_train), "test_size:", len(x_test))

    sym_list = [[0]*390 for _ in range(data_amount)]
    sym_array = np.array(sym_list)
    herb_list = [[0]*805 for _ in range(data_amount)]
    herb_array = np.array(herb_list)

    for i in range(data_amount):
        # sym
        sym_indexes = list(map(int, eval(tcm_lung.iloc[i, 0])))
        sym_indexes = [x for x in sym_indexes if x < 390]
        sym_array[i, sym_indexes] = 1
        # herb
        herb_indexes = list(map(int, eval(tcm_lung.iloc[i, 1])))
        herb_indexes = [x-390 for x in herb_indexes if 390 <= x < 1195]
        herb_array[i, herb_indexes] = 1

    train_dataset = PreDatasetPTM2(sym_array[x_train], herb_array[x_train])
    test_dataset = PreDatasetPTM2(sym_array[x_test], herb_array[x_test])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return x_train, x_test, train_loader, test_loader, [390, 805]


def process_dataset_lung(config, seed, device):
    data_file = 'data/TCM_Lung.xlsx'
    tcm_lung = pd.read_excel(data_file)
    data_amount = tcm_lung.shape[0]

    id_list = [x for x in range(data_amount)]
    x_train, x_test = train_test_split(
        id_list, test_size=config.test_ratio, shuffle=False, random_state=seed
    )

    sym_ids = pd.read_excel(data_file, sheet_name='Symptom Dictionary')
    syn_ids = pd.read_excel(data_file, sheet_name='Syndrome Dictionary')
    treat_ids = pd.read_excel(data_file, sheet_name='Treat Dictionary')
    herb_ids = pd.read_excel(data_file, sheet_name='Herb Dictionary')

    sym_len, syn_len, herb_len, treat_len = len(sym_ids), len(syn_ids), len(herb_ids), len(treat_ids)

    sym_list = [[0]*sym_len for _ in range(data_amount)]
    syn_list = [[0]*syn_len for _ in range(data_amount)]
    herb_list = [[0]*herb_len for _ in range(data_amount)]
    treat_list = [[0]*treat_len for _ in range(data_amount)]

    for i in range(data_amount):
        sym_indexes = list(map(int, tcm_lung.iloc[i, 0].split(',')))
        sym_list[i] = [1 if j in sym_indexes else 0 for j in range(sym_len)]

        syn_indexes = list(map(int, tcm_lung.iloc[i, 1].split(',')))
        syn_list[i] = [1 if j in syn_indexes else 0 for j in range(syn_len)]

        treat_indexes = list(map(int, tcm_lung.iloc[i, 2].split(',')))
        treat_list[i] = [1 if j in treat_indexes else 0 for j in range(treat_len)]

        herb_indexes = list(map(int, tcm_lung.iloc[i, 3].split(',')))
        herb_list[i] = [1 if j in herb_indexes else 0 for j in range(herb_len)]

    sym_array = torch.tensor(np.array(sym_list), device=device, dtype=torch.float32)
    syn_array = torch.tensor(np.array(syn_list), device=device, dtype=torch.float32)
    herb_array = torch.tensor(np.array(herb_list), device=device, dtype=torch.float32)
    treat_array = torch.tensor(np.array(treat_list), device=device, dtype=torch.float32)

    train_dataset = PreDatasetLung4(sym_array[x_train], syn_array[x_train], treat_array[x_train], herb_array[x_train])
    test_dataset = PreDatasetLung4(sym_array[x_test], syn_array[x_test], treat_array[x_test], herb_array[x_test])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)

    return x_train, x_test, train_loader, test_loader, [sym_len, syn_len, treat_len, herb_len]
