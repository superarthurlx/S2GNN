import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
import pdb

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def link_to_onehot(mat, granularity):
    B, _, N = mat.shape # B, 1, N
    # 创建一个形状为 (64, 20, 170) 的全零张量
    one_hot_tensor = torch.zeros(B, granularity, N).to(mat.device)

    # 使用 scatter 将原始张量的值分散到 one-hot 编码张量中
    # shape:(batch, granularity+1, num node)
    return one_hot_tensor.scatter_(1, mat.type(torch.LongTensor).to(mat.device), 1)
    # return one_hot_tensor.scatter_(1, mat[:,-1,:].type(torch.LongTensor).unsqueeze(1).to(mat.device), 1)

# def generate_adjacent_matrix(adj, adj_emb, adj_node):
#     # | E*E, E*N |
#     # | N*E, N*N |
#     B, E, N = adj.shape
#     # adj_mat = torch.zeros(B, E+N, E+N).to(adj.device)
#     adj_mat_up = torch.cat([adj_emb.unsqueeze(0).expand(B, -1, -1), adj], dim=-1)
#     adj_mat_bottom = torch.cat([adj.transpose(1,2), adj_node.unsqueeze(0).expand(B, -1, -1)], dim=-1)
#     adj_mat = torch.cat([adj_mat_up, adj_mat_bottom], dim=1)
#     return adj_mat

def generate_adjacent_matrix(adj):
    """
    A = A_sys + A_rs + A_cs
    """
    # N, _ = adj.shape

    A_sys = sys_adj(adj).unsqueeze(0)
    # A_rs = row_stochastic_adj(adj).unsqueeze(0)
    # A_cs = col_stochastic_adj(adj).unsqueeze(0)

    # adj_mat = torch.cat([A_sys, A_rs, A_cs], dim=0) # (3, N, N)
    # return adj_mat
    return A_sys

def normalized_adj_for_gcn(adj: torch.tensor) -> torch.tensor:
    """Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return A D^{-1}

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """
    # adj = 0.5 * (adj + adj.transpose(1,2)) 
    # add self loop
    adj = adj + torch.diag(torch.ones(adj.shape[1], dtype=torch.float32)).to(adj.device)
    # print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    row_sum = adj.sum(1)
    d_inv = torch.pow(row_sum, -1)
    d_inv = torch.nan_to_num(d_inv, posinf=0)
    d_mat_inv = torch.diag_embed(d_inv)
    mp_adj = adj.bmm(d_mat_inv) 
    return mp_adj

def col_stochastic_adj(adj: torch.tensor) -> torch.tensor:
    """Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return A D^{-1}

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """
    # adj = 0.5 * (adj + adj.transpose(1,2)) 
    # add self loop
    # adj by cosine similairity has already add self loop
    # adj = adj + torch.diag(torch.ones(adj.shape[1], dtype=torch.float32)).to(adj.device)
    # print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    row_sum = adj.sum(1)
    d_inv = torch.pow(row_sum, -1)
    d_inv = torch.nan_to_num(d_inv, posinf=0)
    d_mat_inv = torch.diag_embed(d_inv)
    # mp_adj = adj.mm(d_mat_inv) 
    mp_adj = torch.matmul(adj,d_mat_inv) 
    return mp_adj

def sys_adj(adj: torch.tensor) -> torch.tensor:
    """Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return D^{-1/2} A D^{-1/2}

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """
    # adj = 0.5 * (adj + adj.transpose(1,2)) 
    row_sum = adj.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt = torch.nan_to_num(d_inv_sqrt, posinf=0)
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    mp_adj = torch.matmul(d_mat_inv_sqrt, torch.matmul(adj, d_mat_inv_sqrt))
    return mp_adj

def row_stochastic_adj(adj: torch.tensor) -> torch.tensor:
    """Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return D^{-1} A

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """
    # adj = 0.5 * (adj + adj.transpose(1,2)) 
    # add self loop
    row_sum = adj.sum(1)
    d_inv = torch.pow(row_sum, -1)
    d_inv = torch.nan_to_num(d_inv, posinf=0)
    d_mat_inv = torch.diag_embed(d_inv)
    mp_adj = d_mat_inv.matmul(adj) 
    return mp_adj

# 定义伯恩斯坦多项式逼近函数
def bernstein_approximation(sequence, n):
    def bernstein_basis(k, n, x):
        return comb(n, k) * (x ** k) * ((1 - x) ** (n - k))
    
    def bernstein_poly(sequence, n, x):
        return sum(sequence[k] * bernstein_basis(k, n, x) for k in range(n + 1))
    
    return bernstein_poly(sequence, n, torch.linspace(0, 1, len(sequence)))

def log_comb(n, k):
    """
    Compute the log of the binomial coefficient using log properties.
    This prevents overflow by using logarithms instead of direct computation.
    """
    if k > n:
        return float('-inf')
    if k == 0 or k == n:
        return 0
    return sum(np.log(i) for i in range(n, n-k, -1)) - sum(np.log(i) for i in range(1, k+1))

def bernstein_approximation_log(sequence, n):
    def bernstein_basis(k, n, x):
        log_binom = log_comb(n, k)
        log_term = log_binom + k * np.log(x + 1e-10) + (n - k) * np.log(1 - x + 1e-10)
        return np.exp(log_term)

    def bernstein_poly(sequence, n, x):
        return sum(sequence[k] * bernstein_basis(k, n, x) for k in range(n + 1))
    
    # x_values = torch.linspace(0, 1, len(sequence)).numpy()
    # return [bernstein_poly(sequence, n, x) for x in x_values]
    return bernstein_poly(sequence, n, torch.linspace(0, 1, len(sequence)))
