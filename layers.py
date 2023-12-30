import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ScaledDotProductAttention_hyper(nn.Module):
    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = dropout

    def forward(self, q, k, v, mask):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.dropout(F.softmax(attn, dim=-1), self.dropout, training=self.training)
        output = torch.matmul(attn, v)
        return output


class HyperGraphAttentionLayerSparse(nn.Module):
    def __init__(self, in_features, out_features, dropout, transfer):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)
        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))
        self.word_context = nn.Embedding(1, self.out_features)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.attention1 = ScaledDotProductAttention_hyper(self.out_features ** 0.5, dropout)
        self.attention2 = ScaledDotProductAttention_hyper(self.out_features ** 0.5, dropout)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        x_att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)

        n_edge = adj.shape[1]  # number of edge
        q1 = self.word_context.weight[0:].view(1, 1, -1).repeat(x.shape[0], n_edge, 1)\
            .view(x.shape[0], n_edge, self.out_features)
        edge = self.attention1(q1, x_att, x, mask=adj)

        edge_att = edge.matmul(self.weight3)
        node = self.attention2(x_att, edge_att, edge, mask=adj.transpose(1, 2))

        return node, edge

