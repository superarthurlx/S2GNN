import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .utils import sys_adj

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

class ChebNetII(nn.Module):
    def __init__(self, K, **kwargs):
        super(ChebNetII, self).__init__()
        
        self.K = K
        self.betas = Parameter(torch.Tensor(self.K+1))
        # self.Init=Init
        self.reset_parameters()

    def reset_parameters(self):
        self.betas.data.fill_(1.0)

        # if self.Init:
        for j in range(self.K+1):
            x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
            self.betas.data[j] = x_j**2

    def propagate(self, x, adj):
        return torch.einsum('mn, bnf -> bmf', adj, x)
        
    def forward(self, x, adj_mat):
        coe_tmp=F.relu(self.betas)
        coe=coe_tmp.clone()
        
        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)


        #L=I-D^(-0.5)AD^(-0.5)
        # (余弦相似度自带一个self loop)
        I = torch.diag(torch.ones(adj_mat.shape[1], dtype=torch.float32)).to(adj_mat.device)
        laplacian = sys_adj(adj_mat) 
        # edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #L_tilde=L-I
        laplacian_tilde = laplacian - I
        # edge_index_tilde, norm_tilde= add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(self.node_dim))

        Tx_0=x
        Tx_1=self.propagate(x, adj_mat)

        out=coe[0]/2*Tx_0+coe[1]*Tx_1

        for i in range(2,self.K+1):
            Tx_2=self.propagate(x, laplacian_tilde)
            Tx_2=2*Tx_2-Tx_0
            out=out+coe[i]*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2
        return out, self.betas

    # def message(self, x_j, norm):
    #     return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)