import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers import *


# class HHGAT(nn.Module):
#     def __init__(self, input_size, n_hid, output_size, step, dropout):
#         super(HHGAT, self).__init__()
#         self.dropout = dropout
#         self.step = step
#         self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, self.dropout, transfer=False)
#         self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, self.dropout, transfer=True)
#
#     def forward(self, x, adj):
#         residual = x
#
#         x, y = self.gat1(x, adj)
#
#         if self.step == 2:
#             x = F.dropout(x, self.dropout, training=self.training)
#             x += residual
#             x, y = self.gat2(x, adj)
#
#         x = F.dropout(x, self.dropout, training=self.training)
#         x += residual
#
#         return x


class HHGAT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, step, dropout):
        super(HHGAT, self).__init__()
        self.dropout = dropout
        self.step = step
        self.GAT = nn.ModuleList()
        self.GAT.append(HyperGraphAttentionLayerSparse(input_size, n_hid, self.dropout, transfer=False))
        for i in range(self.step - 2):
            self.GAT.append(HyperGraphAttentionLayerSparse(n_hid, n_hid, self.dropout, transfer=True))
        self.GAT.append(HyperGraphAttentionLayerSparse(n_hid, output_size, self.dropout, transfer=True))

    def forward(self, x, adj):
        residual = x
        for layer in self.GAT:
            x, y = layer(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
            x += residual
        return x
