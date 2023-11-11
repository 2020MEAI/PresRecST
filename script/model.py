#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  : dx
# @File    : model.py
# @Software: PyCharm
# @Note    :
import torch.nn
import torch


class STHNet(torch.nn.Module):
    def __init__(self, batch_size, embedding_dim):
        super(STHNet, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.sym_embedding = torch.nn.Embedding(1801, self.embedding_dim)
        self.syn_embedding = torch.nn.Embedding(54, self.embedding_dim)
        self.treat_embedding = torch.nn.Embedding(67, self.embedding_dim)
        self.herb_embedding = torch.nn.Embedding(410, self.embedding_dim)
        self.mlp_sym = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mlp_syn_1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mlp_syn_2 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.mlp_treat_1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mlp_treat_2 = torch.nn.Linear(self.embedding_dim*2, self.embedding_dim)
        self.mlp_treat_3 = torch.nn.Linear(self.embedding_dim*3, self.embedding_dim)
        self.relu = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim)

        self.bn_layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.bn_layer2 = torch.nn.Sequential(
            torch.nn.Linear(2*self.embedding_dim, self.embedding_dim),
            torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.bn_layer3 = torch.nn.Sequential(
            torch.nn.Linear(3*self.embedding_dim, self.embedding_dim),
            torch.nn.BatchNorm1d(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def forward(self, symptom_OH):
        # 1. symptom embedding
        get_sym = torch.mm(symptom_OH, self.sym_embedding.weight)
        sym_agg = self.mlp_sym(get_sym)
        sym_agg = self.bn_layer1(sym_agg)

        # 2. syndrome embedding
        judge_syndrome = torch.mm(sym_agg, self.syn_embedding.weight.T)
        get_syn = torch.mm(judge_syndrome, self.syn_embedding.weight)

        cat_sym_syn = torch.cat((sym_agg, get_syn), dim=-1)
        syn_agg = self.bn_layer2(cat_sym_syn)

        # 3. treat embedding
        judge_treatment = torch.mm(syn_agg, self.treat_embedding.weight.T)
        get_treat = torch.mm(judge_treatment, self.treat_embedding.weight)

        cat_syn_treat = torch.cat((get_syn, get_treat), dim=-1)
        treat_agg = self.mlp_treat_2(cat_syn_treat)

        cat_syn_treat = torch.cat((sym_agg, syn_agg, treat_agg), dim=-1)  # ADD C  两个 20 => 64*256
        treat_agg = self.bn_layer3(cat_syn_treat)

        # 4. judge herb
        judge_herb = torch.mm(treat_agg, self.herb_embedding.weight.T)  # 20*64 * 64*410 => 20*410 每个人对每个中药判断

        return judge_syndrome, judge_treatment, judge_herb
