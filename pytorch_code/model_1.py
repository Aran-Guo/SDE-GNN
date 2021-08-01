#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 要定义几层
        self.fc1 = torch.nn.Linear(self.input_size, self.input_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden = self.fc2(relu)
        relu = self.relu(hidden)
        output = self.fc3(relu)
        output = self.sigmoid(output)
        return output

class GRU_X(Module):
    def __init__(self, hidden_size, step=2):  # 输入仅需确定隐状态数和步数
        super(GRU_X, self).__init__()
        self.step = step  # gnn前向传播的步数 default=1
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.gate_size = 3 * hidden_size

        # 计算soft_attention
        # self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.soft_attention = nn.Linear(self.hidden_size, 1, bias=True)

        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))


        self.w_hx_hx_res = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hx_hx_res = Parameter(torch.Tensor(self.hidden_size))

        self.w_hx_h_res = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hx_h_res = Parameter(torch.Tensor(self.hidden_size))

        self.w_hx_n = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_hx_n = Parameter(torch.Tensor(self.hidden_size))

    # dropout函数实现
    def dropout(self, x, level):  # level为神经元丢弃的概率值，在0-1之间
        if level < 0. or level >= 1:
            raise Exception('Dropout level must be in interval [0, 1[.')
        retain_prob = 1. - level
        # 利用binomial函数，生成与x一样的维数向量。
        # 神经元x保留的概率为p，n表示每个神经元参与随机实验的次数，通常为1,。
        # size是神经元总数。
        sample = np.random.binomial(n=1, p=retain_prob, size=x.shape)
        sample = torch.from_numpy(sample).cuda()
        # 生成一个0、1分布的向量，0表示该神经元被丢弃
        # print sample
        x *= sample
        # print x
        x /= retain_prob
        return x

    def GRU_XCell(self, hidden, hidden_g, hidden_x, mask):

        # 计算soft_attention
        # q1 = self.linear_one(hidden)    # 上一层的gnn hidden
        # q2 = self.linear_two(hidden_g)    # lstm学习玩的hidden_rnn, batch_size x seq_length x latent_size
        # q3 = self.linear_three(hidden_x)
        # alpha = self.soft_attention(torch.sigmoid(q1 + q2 + q3))     # batch_size x seq_length x 1

        # dropout
        # hidden = torch.nn.Dropout(0.2)(hidden).cuda()
        # hidden_g = torch.nn.Dropout(0.2)(hidden_g).cuda()
        # hidden_x = torch.nn.Dropout(0.2)(hidden_x).cuda()
        # hidden = self.dropout(hidden, 0.1)
        # hidden_g = self.dropout(hidden_g, 0.1)
        # hidden_x = self.dropout(hidden_x, 0.1)

        gi = F.linear(hidden_g, self.w_ih, self.b_ih) # gi-->(100,5?,300) W×at
        gh = F.linear(hidden, self.w_hh, self.b_hh) # gh-->(100,5?,300) W×Vt-1

        hx_hx_res = F.linear(hidden_x, self.w_hx_hx_res, self.b_hx_hx_res) # 维度不匹配
        hx_n = F.linear(hidden_x, self.w_hx_n, self.b_hx_n)
        hx_h_res = F.linear(hidden, self.w_hx_h_res, self.b_hx_h_res)

        i_r, i_i, i_n = gi.chunk(3, 2)  # 三个都是(100,5?,100)
        h_r, h_i, h_n = gh.chunk(3, 2)  # 三个都是(100,5?,100)

        resetgate = torch.sigmoid(i_r + h_r)   # 重置门
        inputgate = torch.sigmoid(i_i + h_i)   # 更新门
        resgate = torch.sigmoid(hx_h_res + hx_hx_res)   # 残差门

        # 对inputgate加入soft_attention 其中inputgate-->(100,5,100)  alpha-->(100, 5, 1)
        # inputgate = alpha * inputgate


        # dropout
        i_n = torch.nn.Dropout(0.2)(i_n).cuda()
        h_n = torch.nn.Dropout(0.2)(h_n).cuda()
        hx_n = torch.nn.Dropout(0.2)(hx_n).cuda()
        newgate = torch.tanh(i_n + resetgate * h_n + resgate * hx_n)  # newgate-->(100,5?,100)  原文公式(4)
        hy = newgate + inputgate * (hidden - newgate)   # hy-->(100,5?,100)    原文公式(5)
        return hy

    def forward(self,hidden, hidden_g, hidden_x, mask):
        # A-->实际上是该批数据图矩阵的列表 eg:(100,5?,10?(即5?X2)) 5?代表这个维的长度是该批唯一最大类别长度(类别数目不足该长度的会话补零)，根据不同批会变化
        # hidden--> eg:(100-batch_size,5?,100-embeding_size) 即数据图中节点类别对应低维嵌入的表示
        for i in range(self.step):
            hidden = self.GRU_XCell(hidden, hidden_g, hidden_x, mask)
        return hidden


class GNN(Module):
    def __init__(self, hidden_size, step=1):  # 输入仅需确定隐状态数和步数
        super(GNN, self).__init__()
        self.step = step  # gnn前向传播的步数 default=1
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        # 计算soft_attention
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.soft_attention = nn.Linear(self.hidden_size, 1, bias=True)

        # 有关Parameter函数的解释：首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        # 所以经过类型转换这个self.XX变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。——————https://www.jianshu.com/p/d8b77cc02410

        # 模型参数共享
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
        # 有关nn.Linear的解释：torch.nn.Linear(in_features, out_features, bias=True)，对输入数据做线性变换：y=Ax+b
        # 形状：输入: (N,in_features)  输出： (N,out_features)
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # 计算x
        # self.linear_x_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_x_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.b_ixh = Parameter(torch.Tensor(self.hidden_size))
        # self.b_oxh = Parameter(torch.Tensor(self.hidden_size))

        # x的参数
        # self.w_ih_x = Parameter(torch.Tensor(self.hidden_size * 2, self.input_size))
        # self.w_hh_x = Parameter(torch.Tensor(self.hidden_size * 2, self.hidden_size))
        # self.b_ih_x = Parameter(torch.Tensor(self.hidden_size * 2))
        # self.b_hh_x = Parameter(torch.Tensor(self.hidden_size * 2))
    def GNNCell(self, A, hidden, hidden_rnn, hidden_x, mask):
        # A-->实际上是该批数据图矩阵的列表  eg:(100,5?,10?(即5?X2))
        # hidden--> eg(100-batch_size,5?,100-embeding_size)

        # 计算soft_attention
        # ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1] 是否使用mask？
        # q1 = self.linear_one(hidden)    # 上一层的gnn hidden
        # q2 = self.linear_two(hidden_rnn)    # lstm学习玩的hidden_rnn, batch_size x seq_length x latent_size
        # q3 = self.linear_three(hidden_x)
        # alpha = self.soft_attention(torch.sigmoid(q1 + q2 + q3))     # batch_size x seq_length x 1

        # 后面所有的5?代表这个维的长度是该批唯一最大类别长度(类别数目不足该长度的会话补零)，根据不同批会变化
        # 有关matmul的解释：矩阵相乘，多维会广播相乘
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah   # input_in-->(100,5?,100)
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah  # input_out-->(100,5?,100)
        # x_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_x_in(hidden)) + self.b_ixh  # input_in-->(100,5?,100)
        # x_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_x_out(hidden)) + self.b_oxh  # input_out-->(100,5?,100)
        # 在第2个轴将tensor连接起来
        inputs = torch.cat([input_in, input_out], 2)  # inputs-->(100,5?,200)
        # x = torch.cat([x_in, x_out], 2)
        # 关于functional.linear(input, weight, bias=None)的解释：y= xA^T + b 应用线性变换，返回Output: (N,∗,out_features)
        # [*代表任意其他的东西]
        gi = F.linear(inputs, self.w_ih, self.b_ih) # gi-->(100,5?,300) W×at
        gh = F.linear(hidden, self.w_hh, self.b_hh) # gh-->(100,5?,300) W×Vt-1

        # xi = F.linear(x, self.w_ih_x, self.b_ih_x)

        # hx = F.linear(hidden_x, self.w_hr, self.b_hr) # 维度不匹配
        # hxhx = F.linear(hidden_x, self.w_hrhr, self.b_hrhr)
        # hxh = F.linear(hidden, self.w_hrh, self.b_hrh)
        # torch.chunk(tensor, chunks, dim=0)：将tensor拆分成指定数量的块，比如下面就是沿着第2个轴拆分成3块

        i_r, i_i, i_n = gi.chunk(3, 2)  # 三个都是(100,5?,100)
        h_r, h_i, h_n = gh.chunk(3, 2)  # 三个都是(100,5?,100)

        # x_x_gate, hxhx = xi.chunk(2, 2)  # 三个都是(100,5?,100)

        resetgate = torch.sigmoid(i_r + h_r)   # resetgate-->(100,5?,100)      原文公式(3)
        inputgate = torch.sigmoid(i_i + h_i)   # inputgate-->(100,5?,100)
        # xgate = torch.sigmoid(hx + hxh)
        # 对inputgate加入soft_attention 其中inputgate-->(100,5,100)  alpha-->(100, 5, 1)
        # inputgate = alpha * inputgate
        # newgate = torch.tanh(i_n + resetgate * h_n + xgate * hxhx)  # newgate-->(100,5?,100)  原文公式(4)
        newgate = torch.tanh(i_n + resetgate * h_n)  # newgate-->(100,5?,100)  原文公式(4)
        hy = newgate + inputgate * (hidden - newgate)   # hy-->(100,5?,100)    原文公式(5)
        return hy

    def forward(self, A, hidden, hidden_rnn, hidden_x, mask):
        # A-->实际上是该批数据图矩阵的列表 eg:(100,5?,10?(即5?X2)) 5?代表这个维的长度是该批唯一最大类别长度(类别数目不足该长度的会话补零)，根据不同批会变化
        # hidden--> eg:(100-batch_size,5?,100-embeding_size) 即数据图中节点类别对应低维嵌入的表示
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden, hidden_rnn, hidden_x, mask)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node): # opt-->可控输入参数, n_node-->嵌入层图的节点数目
        super(SessionGraph, self).__init__()    # 这一行是什么意思？？？
        self.hidden_size = opt.hiddenSize  # opt.hiddenSize-->hidden state size
        self.n_node = n_node    # 节点的个数，是一个整数
        self.batch_size = opt.batchSize   # opt.batch_siza-->input batch size *default=100
        self.nonhybrid = opt.nonhybrid   # opt.nonhybrid-->only use the global preference to predicts，不考虑短时偏好

        self.embedding_rnn = nn.Embedding(self.n_node, self.hidden_size)    # rnn embedding层
        self.embedding_x = nn.Embedding(self.n_node, self.hidden_size)    # 原始embedding
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)    # graph embedding层， 节点总数 × embedding size

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True).cuda()    # 学习rnn embedding的模型
        self.gnn = GNN(self.hidden_size, step=opt.step)  # opt.step-->gnn propogation steps， 学习graph embedding
        self.lstm_layers = nn.LSTM(self.hidden_size, self.hidden_size // 2, 1, batch_first=True,
                              bidirectional=True).cuda()  # Bi-LSTM

        self.gru_x = GRU_X(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, 2, batch_first = True, dropout = 0.2).cuda()    # 交互层
        self.feedforward = Feedforward(self.hidden_size * 2, self.hidden_size)    # 3层
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)    # 将学习到的rnn_embedding加入线性函数，这个函数是不是太简单了

        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_three2 = nn.Linear(self.hidden_size, 1, bias=False)     # 计算soft attention

        # fusion gate b
        self.linear_gate1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_gate2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fusion_gate1 = nn.Linear(self.hidden_size, 1, bias = True)

        self.linear_gate3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_gate4 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fusion_gate2 = nn.Linear(self.hidden_size, 1, bias = True)

        # 交互层 BILSTM
        self.bilstm_interaction = nn.LSTM(self.hidden_size, self.hidden_size, 1, batch_first=True,
                                   bidirectional=True).cuda()  # Bi-LSTM

        # BN
        self.BN1 = nn.BatchNorm1d(16)
        self.BN2 = nn.BatchNorm1d(16)

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()  # 交叉熵损失，这个输入z会自动的进行softmax得到y再计算
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2) # Adam优化算法
        # StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1) 将每个参数组的学习率设置为每个step_size epoch
        # 由gamma衰减的初始lr。当last_epoch=-1时，将初始lr设置为lr。
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()   # 初始化权重参数

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, hidden_rnn, mask):
        # hidden-->(100,16?,100) 其中16?代表该样本所有数据最长会话的长度(不同数据集会不同)，单个样本其余部分补了0
        # mask-->(100,16?) 有序列的位置是[1],没有动作序列的位置是[0]
        # hidden = hidden + hidden_rnn    # 成功，提高大概1个点
        # hidden = self.BN1(hidden)
        # hidden_rnn = self.BN1(hidden_rnn)
        # Q1 = self.linear_gate1(hidden)
        # Q2 = self.linear_gate2(hidden_rnn)
        # beta = self.fusion_gate1(torch.sigmoid(Q1 + Q2))
        # hidden = (1 - beta) * hidden + beta * hidden_rnn
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size(100,100) 这是最后一个动作对应的位置，即文章中说的局部偏好
        # ht_rnn = hidden_rnn[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]


        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size(100,1,100) 局部偏好线性变换后改成能计算的维度(和全局偏好一个维度)
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size (100,16?,100) 即全局偏好
        # q2 = q2.cuda()
        # q2 = self.lstm_layers(q2)[0]   # lstm_layers会返回一个元组类型
        # hidden_rnn = self.linear_two2(hidden_rnn)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (100,16,1)， 计算软注意力
        # beta = self.linear_three2(torch.sigmoid(q3))

        # a_rnn = torch.sum(beta * hidden_rnn * mask.view(mask.shape[0], -1, 1).float(), 1)   # batch_size × hidden_size
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1) # (100,100)  原文中公式(6)

        # hidden_rnn
        # hidden_rnn = hidden_rnn * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size

        if not self.nonhybrid:
            # 加入rnn_embedding信息并加入MLP
            # a = self.feedforward(torch.cat([a, a_rnn, ht], 1))  # 原文中公式(7) batch_size × hidden_size
            a = self.linear_transform(torch.cat([a, ht], 1))  # 原文中公式(7)

        # the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from \mathcal{N}(0, 1)N(0,1)
        b = self.embedding.weight[1:]  # n_nodes x latent_size  (309,100) b是每个节点的embedding(target embedding)
        # b_rnn = self.embedding.weight[1:]

        # hidden_rnn target attention: sigmoid(hidden M b)
        # mask  # batch_size x seq_length
        # hidden_rnn = hidden_rnn * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size
        # qt_rnn = self.linear_t(hidden_rnn)  # batch_size x seq_length x latent_size
        # beta = torch.sigmoid(b @ qt.transpose(1,2))  # batch_size x n_nodes x seq_length
        # beta = F.softmax(b_rnn @ qt_rnn.transpose(1,2), -1)  # batch_size x n_nodes x seq_length
        # hidden_rnn = beta @ hidden  # batch_size x n_nodes x latent_size

        scores = torch.matmul(a, b.transpose(1, 0))   # 原文中公式(8)，transpose是专职 +
        return scores  # (100,309)

    def forward(self, inputs, A, mask):  # items,A
        # inputs-->单个点击动作序列的唯一类别并按照批最大唯一类别长度补全0列表(即图矩阵的元素的类别标签列表)  A-->实际上是该批数据图矩阵的列表
#        print(inputs.size())  #测试打印下输入的维度  （100-batch_size,5?） 5?代表这个维的长度是该批唯一最大类别长度(类别数目不足该长度的会话补0)，根据不同批会变化
        hidden = self.embedding(inputs) # 返回的hidden的shape -->（100-batch_size,5?,100-embeding_size）
        hidden_x = self.embedding(inputs)
        hidden_rnn = self.embedding(inputs)
        # 计算soft-attention函数，传入gnn中
        hidden_rnn = self.lstm(hidden_rnn)[0]
        hidden = self.gnn(A, hidden, hidden_rnn, hidden_x, mask)
        hidden = hidden.cuda()

        # 交互层
        # hidden = self.gru_x(hidden, hidden, hidden_x, mask)
        # hidden = self.bilstm_interaction(hidden)[0]
        # hidden = self.gru(hidden)[0]

        # fusion gate
        # Q3 = self.linear_gate3(hidden[:, :, :100])
        # Q3 = Q3.cuda()
        # Q4 = self.linear_gate4(hidden[:, :, 100:])
        # Q4 = Q4.cuda()
        # beta = self.fusion_gate2(torch.sigmoid(Q3 + Q4))
        # hidden = (1 - beta) * hidden[:, :, :100] + beta * hidden[:, :, 100:]

        # feedforward
        # hidden = self.feedforward(hidden)

        return hidden, hidden_rnn  # (100,5?,100)


def trans_to_cuda(variable):    # 这里是，SessionGraph。 传入torch.Tensor(alias_inputs).long()
    if torch.cuda.is_available():
        return variable.cuda()  # 返回该对象在gpu的版本
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):  # 传入模型model(SessionGraph), 数据批的索引i, 训练的数据data(Data)
    # 返回：动作序列对应唯一动作集合的位置角标，该批数据图矩阵的列表，单个点击动作序列的唯一类别并按照批最大类别补全0列表，面罩，目标数据
    alias_inputs, A, items, mask, targets = data.get_slice(i)  # alias_inputs记录不同商品的index，A是连接矩阵，items记录同步商品
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())  #(100,16?)
    # test_alias_inputs = alias_inputs.numpy()  #测试查看alias_inputs的内容
    # strange = torch.arange(len(alias_inputs)).long() #0到99
    items = trans_to_cuda(torch.Tensor(items).long())   # 都转换为long
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden, hidden_rnn = model(items, A, mask)  # 这里调用了SessionGraph的forward函数,返回维度数目(100,5?,100)
    # get函数要理解
    get = lambda i: hidden[i][alias_inputs[i]]   # 选择第这一批第i个样本对应类别序列的函数
    get_rnn = lambda i: hidden_rnn[i][alias_inputs[i]]  # 选择第这一批第i个样本对应类别序列的函数
    # test_get = get(0)  # (16?,100) 测试用的
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # (100,16?,100)
    seq_hidden_rnn = torch.stack([get_rnn(i) for i in torch.arange(len(alias_inputs)).long()])  # (100,16?,100)
    return targets, model.compute_scores(seq_hidden, seq_hidden_rnn, mask)

# 计算方法
def mrrAtK(pScore, nScore, atK=None):
    """
    MRR@
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """
    pScore = np.asarray(pScore).flatten()
    nScore = np.asarray(nScore).flatten()

    T = len(pScore)

    if atK is None:
        atK = T

    mrr = np.zeros_like(atK, dtype=float)
    for p in pScore:
        rank = np.sum(nScore > p) + 1
        mrr += (rank <= atK) * (1 / rank)

    return mrr / T

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


def auc_scores(scores, targets, k=None):
    if k == None:
        k = len(scores[0])
    sub_scores = scores.topk(k)[1]  # scores是概率分布，sub_scores是预测标签
    sub_scores = trans_to_cpu(sub_scores).detach().numpy()
    aucs = []
    for score, target in zip(sub_scores, targets):
        auc = roc_auc_score(target, score)
        aucs.append(auc)
    auc_score = np.mean(aucs) * 100
    return auc_score


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    # for target, score in zip(targets, scores):
    #     best = dcg_score(target, target, k)
    #     actual = dcg_score(target, score, k)
    y_score = trans_to_cpu(y_score).detach().numpy()
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def train_test(model, train_data, test_data): # 传入模型SessionGraph，训练数据和测试数据Data
    model.scheduler.step()  # 调度设置优化器的参数，就是调节学习率，经过多少步缩小一次学习率。https://blog.csdn.net/qq_20622615/article/details/83150963
    print('start training: ', datetime.datetime.now())
    model.train()  # 指定模型为训练模式，计算梯度
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) # 得到批的索引，是一个二维数组，每一行表示第几个batch，每一个元素代表一个样本的索引
    for i, j in zip(slices, np.arange(len(slices))):   # 根据批的索引数据进行数据提取训练:i-->批索引, j-->第几批
        model.optimizer.zero_grad()  # 前一步的损失清零
        targets, scores = forward(model, i, train_data) # 将第i批传入模型
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1) # 交叉熵损失，targets是真值
        loss.backward() # 反向传播
        model.optimizer.step()  # 优化
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()  # 指定模型为计算模式
    # hit, mrr = [], []   # hit是P，mrr是MRR
    slices = test_data.generate_batch(model.batch_size)

    hits_20 = []
    mrrs_20 = []
    hits_10 = []
    mrrs_10 = []
    hits_5 = []
    mrrs_5 = []
    ndcg5s = []
    ndcg10s = []
    for i in slices:
        targets, scores = forward(model, i, test_data)
        hits_20, mrrs_20 = hit_mrr_n(targets, scores, hits_20, mrrs_20, 20)
        hits_10, mrrs_10 = hit_mrr_n(targets, scores, hits_10, mrrs_10,10)
        hits_5, mrrs_5 = hit_mrr_n(targets, scores, hits_5, mrrs_5, 5)
        # ndcg_20 = ndcg_score(targets, scores, 20)
        # aucs = roc_auc_score(targets, scores)
        # auc = np.mean(aucs)
        # auc = auc_scores(scores, targets)
        # auc_20 = auc_scores(scores, targets, 20)
        # auc_10 = auc_scores(scores, targets, 10)
        # auc_5 = auc_scores(targets, scores, 5)
        # sub_scores = scores.topk(20)[1] # scores是概率分布，sub_scores是预测标签
        # sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        # for score, target, mask in zip(sub_scores, targets, test_data.mask):    # score是TopN数组，target是标签常量
        #     hit.append(np.isin(target - 1, score))    # 预测的标签在top N里则为1.
        #     if len(np.where(score == target - 1)[0]) == 0:
        #         mrr.append(0)
        #     else:
        #         mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hitm_20 = np.mean(hits_20) * 100
    mrrm_20 = np.mean(mrrs_20) * 100
    hitm_10 = np.mean(hits_10) * 100
    mrrm_10 = np.mean(mrrs_10) * 100
    hitm_5 = np.mean(hits_5) * 100
    mrrm_5 = np.mean(mrrs_5) * 100
    return hitm_20, hitm_10, hitm_5, \
           mrrm_20, mrrm_10, mrrm_5, \
