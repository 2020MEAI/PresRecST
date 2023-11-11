#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import pandas as pd


def training_and_testing(model, x_train, x_test, train_loader, test_loader, params, optimizer, criterion, scheduler):
    train_loss_list = []
    test_loss_list = []

    for epoch in range(params.epoch):
        model.train()
        training_loss = 0.0
        for i, (sid, sdid, tid, hid) in enumerate(train_loader):
            sid, sdid, tid, hid = sid.float(), sdid.float(), tid.float(), hid.float()
            optimizer.zero_grad()
            pre_syn, pre_treat, pre_herb = model(sid)
            loss = params.alpha1 * criterion(pre_herb, hid) + params.alpha2 * criterion(pre_syn, sdid)\
                   + params.alpha3 * criterion(pre_treat, tid)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        print('[Epoch {}]train_loss: '.format(epoch + 1), training_loss / len(train_loader), end='\t')
        train_loss_list.append(training_loss / len(train_loader))

        model.eval()
        test_loss = 0
        test_p5 = 0
        test_p10 = 0
        test_p20 = 0
        test_r5 = 0
        test_r10 = 0
        test_r20 = 0
        test_f1_5 = 0
        test_f1_10 = 0
        test_f1_20 = 0

        for i, (ttsid, ttsdid, tttid, tthid) in enumerate(test_loader):
            ttsid, ttsdid, tttid, tthid = ttsid.float(), ttsdid.float(), tttid.float(), tthid.float()
            ttpre_syn, ttpre_treat, ttpre_herb = model(ttsid)
            temp_ttloss = params.alpha1 * criterion(ttpre_herb, tthid) + params.alpha2 * criterion(ttpre_syn, ttsdid)\
                          + params.alpha3 * criterion(ttpre_treat, tttid)
            test_loss += temp_ttloss.item()

            for i, hid in enumerate(tthid):
                trueLabel = []  # 对应存在草药的索引
                for idx, val in enumerate(hid):  # 获得thid中值为一的索引
                    if val == 1:
                        trueLabel.append(idx)

                top5 = torch.topk(ttpre_herb[i], 5)[1]  # 预测值前5索引
                count = 0
                for m in top5:
                    if m in trueLabel:
                        count += 1
                test_p5 += count / 5
                test_r5 += count / len(trueLabel)

                top10 = torch.topk(ttpre_herb[i], 10)[1]  # 预测值前10索引
                count = 0
                for m in top10:
                    if m in trueLabel:
                        count += 1
                test_p10 += count / 10
                test_r10 += count / len(trueLabel)

                top20 = torch.topk(ttpre_herb[i], 20)[1]  # 预测值前20索引
                count = 0
                for m in top20:
                    if m in trueLabel:
                        count += 1
                test_p20 += count / 20
                test_r20 += count / len(trueLabel)

        test_loss_list.append(test_loss / len(test_loader))
        print('test_loss: '.format(epoch + 1), test_loss / len(test_loader))
        scheduler.step()

    test_loss = 0
    topk_p_count = [0 for _ in range(20)]
    topk_r_count = [0 for _ in range(20)]

    print('-- HR result --')
    for i, (ttsid, ttsdid, tttid, tthid) in enumerate(test_loader):
        ttsid, ttsdid, tttid, tthid = ttsid.float(), ttsdid.float(), tttid.float(), tthid.float()
        ttpre_syn, ttpre_treat, ttpre_herb = model(ttsid)
        temp_ttloss = params.alpha1 * criterion(ttpre_herb, tthid) + params.alpha2 * criterion(ttpre_syn, ttsdid) \
                      + params.alpha3 * criterion(
            ttpre_treat, tttid)
        test_loss += temp_ttloss

        for i, hid in enumerate(tthid):
            trueLabel = []  # 对应存在草药的索引
            for idx, val in enumerate(hid):  # 获得thid中值为一的索引
                if val == 1:
                    trueLabel.append(idx)

            # record topk indexes, for k in range 1~20
            top20_list = [torch.topk(ttpre_herb[i], cnt)[1] for cnt in range(1, 21)]

            for cnt in range(20):
                count = 0
                for pre in top20_list[cnt]:
                    if pre in trueLabel:
                        count += 1
                topk_p_count[cnt] += count / (cnt + 1)
                topk_r_count[cnt] += count / len(trueLabel)

    print("----------------------------------------------------------------------------------------------------")
    print('test_loss: ', test_loss / len(test_loader))
    print('p5-10-20:', topk_p_count[4] / len(x_test), topk_p_count[9] / len(x_test), topk_p_count[19] / len(x_test))
    print('r5-10-20:', topk_r_count[4] / len(x_test), topk_r_count[9] / len(x_test), topk_r_count[19] / len(x_test))
    print('f1_5-10-20: ',
          2 * (topk_p_count[4] / len(x_test)) * (topk_r_count[4] / len(x_test)) / (
                      (topk_p_count[4] / len(x_test)) + (topk_r_count[4] / len(x_test))),
          2 * (topk_p_count[9] / len(x_test)) * (topk_r_count[9] / len(x_test)) / (
                      (topk_p_count[9] / len(x_test)) + (topk_r_count[9] / len(x_test))),
          2 * (topk_p_count[19] / len(x_test)) * (topk_r_count[19] / len(x_test)) / (
                      (topk_p_count[19] / len(x_test)) + (topk_r_count[19] / len(x_test))))

    test_df = pd.DataFrame()
    p_test = []
    r_test = []
    f1_test = []

    for i in range(20):
        p_test.append(topk_p_count[i] / len(x_test))
        r_test.append(topk_r_count[i] / len(x_test))
        f1_test.append(2 * p_test[i] * r_test[i] / (p_test[i] + r_test[i]))

    test_df['K'] = [i + 1 for i in range(20)]
    test_df['Precision'] = p_test
    test_df['Recall'] = r_test
    test_df['F1-score'] = f1_test
    test_df.to_excel("result/herb_result.xlsx", index=False)
