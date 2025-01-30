import numpy as np
import torch

import pdb

def masked_mase(prediction: torch.Tensor, target: torch.Tensor, seasonal_period: int = 1, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Scaled Error (MASE) between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    A scaled error is less than one if it arises from a better forecast than the average one-step naive
    forecast computed in-sample. Conversely, it is greater than one if the forecast is worse than the
    average one-step naive forecast computed in-sample.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        seasonal_period (int): The seasonal period of the data.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute scaled error.
    """

    if len(prediction.shape) == 4: # (Bs, L, N, 1) else (Bs, N, 1)
        prediction = torch.mean(prediction, dim=1)
        target = torch.mean(target, dim=1)

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    target = target * mask  # Apply the mask to the target
    target = torch.nan_to_num(target)  # Replace any NaNs in the target with zero

    # Calculate the naive forecast error
    naive_forecast = target[seasonal_period:]
    naive_target = target[:-seasonal_period]
    naive_error = torch.abs(naive_forecast - naive_target)
    naive_error = naive_error * mask[seasonal_period:]  # Apply the mask to the naive error
    naive_error = torch.nan_to_num(naive_error)  # Replace any NaNs in the naive error with zero
    n = prediction.shape[0]  # Number of samples
    mase = loss / (torch.sum(naive_error) / (n - seasonal_period))
    return torch.mean(mase)