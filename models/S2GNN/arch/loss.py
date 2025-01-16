import torch
import numpy as np
from basicts.losses import masked_mae
import torch.nn.functional as F
import pdb

def DualExpCLR(embedding, temperature=0.5): 
    N, _ = embedding.shape
    embedding_norm = embedding / torch.norm(embedding, dim=-1, keepdim=True)
    cl = torch.matmul(embedding_norm, embedding_norm.t()) 
    cl = cl.relu()
    cl_p = cl - torch.diag_embed(torch.diag(cl))
    # cl_n = 1 - cl
    cl_p = temperature * (torch.exp(cl_p / temperature)**2 / N**2).sum().log()
    # cl_n = temperature * (torch.exp(cl_n / temperature)**2 / N**2).sum().log()
    return cl_p #+ cl_n

def s2gnn_loss(prediction, target, embedding, temperature, null_val = np.nan):
    # regression loss
    loss_e = DualExpCLR(embedding, temperature)
    loss_r = masked_mae(prediction, target, null_val=null_val)
    # total loss
    loss = loss_e + loss_r
    return loss

# def masked_mae(prediction: torch.Tensor, target: torch.Tensor, hist_avg, hist_std, null_val: float = np.nan) -> torch.Tensor:
#     """Masked mean absolute error.

#     Args:
#         prediction (torch.Tensor): predicted values
#         target (torch.Tensor): labels
#         null_val (float, optional): null value. Defaults to np.nan.

#     Returns:
#         torch.Tensor: masked mean absolute error
#     """

#     if np.isnan(null_val):
#         mask = ~torch.isnan(target)
#     else:
#         eps = 5e-5
#         mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
#     mask = mask.float()
#     mask /= torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(prediction-target)
#     loss = loss * mask

#     hist_avg = torch.exp(-1/hist_avg).to(loss.device)
#     hist_std = torch.exp(-1/hist_std).to(loss.device)
#     loss = loss * hist_std + hist_avg
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)