from typing import Optional
import math
import torch
import torch.nn as nn 
from torch.nn import Parameter
from scipy.special import comb
import torch.nn.functional as F
import numpy as np
from .utils import sys_adj

import pdb

class BernNet(nn.Module):
    def __init__(self, K, **kwargs):
        super(BernNet, self).__init__()
        
        self.K = K
        self.betas = Parameter(torch.ones(self.K+1), requires_grad=True)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.betas.data.fill_(1)

    def propagate(self, x, adj):
        return torch.einsum('mn, bnf -> bmf', adj, x)

    def forward(self, x, adj_mat):
        TEMP=F.relu(self.betas)

        #L=I-D^(-0.5)AD^(-0.5)
        # (余弦相似度自带一个self loop)
        I = torch.diag(torch.ones(adj_mat.shape[1], dtype=torch.float32)).to(adj_mat.device)
        laplacian = sys_adj(adj_mat) - I
        # edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        #2I-L 
        norm_laplacian = 2*I - laplacian
        # edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(x, norm_laplacian)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
            x=tmp[self.K-i-1]
            x=self.propagate(x, laplacian)
            for j in range(i):
                x=self.propagate(x, laplacian)

            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x

        return out, self.betas

    # def message(self, x_j, norm):
    #     return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.betas)