from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import pdb


def l1_loss(prediction: torch.Tensor, target: torch._tensor, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = "mean") -> torch.Tensor:
    """unmasked mae."""

    return F.l1_loss(prediction, target, size_average=size_average, reduce=reduce, reduction=reduction)


def l2_loss(prediction: torch.Tensor, target: torch.Tensor, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = "mean") -> torch.Tensor:
    """unmasked mse"""

    return F.mse_loss(prediction, target, size_average=size_average, reduce=reduce, reduction=reduction)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(prediction-target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean squared error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean squared error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (prediction-target)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """root mean squared error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value . Defaults to np.nan.

    Returns:
        torch.Tensor: root mean squared error
    """

    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))


def masked_mape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked mean absolute percentage error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    target = torch.where(torch.abs(target) < 1e-4, torch.zeros_like(target), target)
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.abs(prediction-target)/target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def dual_loss(prediction: torch.Tensor, target: torch.Tensor, cl_loss: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    loss = masked_mae(prediction, target, null_val=null_val)
    # mse
    # cl_loss = cl_loss ** 2
    cl_loss = torch.where(torch.isnan(cl_loss), torch.zeros_like(cl_loss), cl_loss)
    return loss + torch.mean(cl_loss)


def link_pred_mae(link_pred: torch.Tensor, link_target: torch.Tensor):
    '''
    shape, (K, B, num_class, Node) or (K, B, Node)
    混淆矩阵example:
        true_labels = torch.tensor([0, 1, 0, 2, 1, 0])
        predicted_labels = torch.tensor([0, 1, 1, 2, 1, 0])

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(true_labels.numpy(), predicted_labels.numpy())

    '''
    if len(link_target.shape) == 4:
        K, B, _, N = link_target.shape
        link_pred = link_pred.argmax(2)
        link_target = link_target.argmax(2)
    else:
        K, B, N = link_pred.shape
    return (link_pred == link_target).sum() / (K * B * N)

def link_input_mae(link_target: torch.Tensor, link_input: torch.Tensor):
    if len(link_target.shape) == 4:
        K, B, _, N = link_target.shape
        link_target = link_target.argmax(2)
        link_input = link_input.argmax(2)
    else:
        K, B, N = link_target.shape
    return (link_input == link_target).sum() / (K * B * N)