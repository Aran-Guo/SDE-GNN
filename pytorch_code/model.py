#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.w_hr = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w_hrh = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hr = Parameter(torch.Tensor(self.hidden_size))
        self.b_hrh = Parameter(torch.Tensor(self.hidden_size))
        self.w_hrhr = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hrhr = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # soft_attention 计算
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.soft_attention = nn.Linear(self.hidden_size, 1, bias=True)

    def GNNCell(self, A, hidden, hidden_rnn, hidden_x):

        # 计算soft_attention
        q1 = self.linear_one(hidden)    # 上一层的gnn hidden
        q2 = self.linear_two(hidden_rnn)    # lstm学习玩的hidden_rnn, batch_size x seq_length x latent_size
        q3 = self.linear_three(hidden_x)
        alpha = self.soft_attention(torch.sigmoid(q1 + q2 + q3))     # batch_size x seq_length x 1

        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)

        hx = F.linear(hidden_x, self.w_hr, self.b_hr) # 维度不匹配
        hxhx = F.linear(hidden_x, self.w_hrhr, self.b_hrhr)
        hxh = F.linear(hidden, self.w_hrh, self.b_hrh)

        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        inputgate = alpha * inputgate
        xgate = torch.sigmoid(hx + hxh)
        # newgate = torch.tanh(i_n + resetgate * h_n)
        newgate = torch.tanh(i_n + resetgate * h_n + xgate * hxhx)  # newgate-->(100,5?,100)  原文公式(4)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden, hidden_rnn, hidden_x):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden, hidden_rnn, hidden_x)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True).cuda()  # 学习rnn embedding的模型

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)

        # BN
        self.BN1 = nn.BatchNorm1d(self.hidden_size)
        self.BN2 = nn.BatchNorm1d(self.hidden_size)

        # fusion gate b
        self.linear_gate1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_gate2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fusion_gate1 = nn.Linear(self.hidden_size, 1, bias = True)

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, hidden_rnn, mask):

        Q1 = self.linear_gate1(hidden)
        Q2 = self.linear_gate2(hidden_rnn)
        beta = self.fusion_gate1(torch.sigmoid(Q1 + Q2))
        hidden = (1 - beta) * hidden + beta * hidden_rnn

        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A, mask):
        hidden = self.embedding(inputs)
        hidden_x = self.embedding(inputs)
        hidden_rnn = self.embedding(inputs)
        hidden_rnn = self.lstm(hidden_rnn)[0]
        hidden = self.gnn(A, hidden, hidden_rnn, hidden_x)
        # hidden = hidden.cuda()

        return hidden, hidden_rnn


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden, hidden_rnn = model(items, A, mask)  # 这里调用了SessionGraph的forward函数,返回维度数目(100,5?,100)
    get = lambda i: hidden[i][alias_inputs[i]]  # 选择第这一批第i个样本对应类别序列的函数
    get_rnn = lambda i: hidden_rnn[i][alias_inputs[i]]  # 选择第这一批第i个样本对应类别序列的函数
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # (100,16?,100)
    seq_hidden_rnn = torch.stack([get_rnn(i) for i in torch.arange(len(alias_inputs)).long()])  # (100,16?,100)
    return targets, model.compute_scores(seq_hidden, seq_hidden_rnn, mask)

def hit_mrr_n(targets, scores, hits_n, mrrs_n, k):
    sub_scores = scores.topk(k)[1]  # scores是概率分布，sub_scores是预测标签
    sub_scores = trans_to_cpu(sub_scores).detach().numpy()
    for score, target in zip(sub_scores, targets):  # score是TopN数组，target是标签常量
        # hits_n.append(np.isin(target - 1, score))  # 预测的标签在top N里则为1.
        hits_n.append(np.isin(target - 1, score))
        if len(np.where(score == target - 1)[0]) == 0:
            mrrs_n.append(0)
        else:
            mrrs_n.append(1 / (np.where(score == target - 1)[0][0] + 1))

    return hits_n, mrrs_n

def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hits_20 = []
    mrrs_20 = []
    hits_10 = []
    mrrs_10 = []
    hits_5 = []
    mrrs_5 = []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        hits_20, mrrs_20 = hit_mrr_n(targets, scores, hits_20, mrrs_20, 20)
        hits_10, mrrs_10 = hit_mrr_n(targets, scores, hits_10, mrrs_10, 10)
        hits_5, mrrs_5 = hit_mrr_n(targets, scores, hits_5, mrrs_5, 5)

    hitm_20 = np.mean(hits_20) * 100
    mrrm_20 = np.mean(mrrs_20) * 100
    hitm_10 = np.mean(hits_10) * 100
    mrrm_10 = np.mean(mrrs_10) * 100
    hitm_5 = np.mean(hits_5) * 100
    mrrm_5 = np.mean(mrrs_5) * 100
    return hitm_20, hitm_10, hitm_5, \
           mrrm_20, mrrm_10, mrrm_5, \
