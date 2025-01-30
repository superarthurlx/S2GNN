import numpy as np
import torch


def masked_rmsse(prediction: torch.Tensor, target: torch.Tensor, seasonal_period: int = 1, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the MaskedRoot Mean Squared Scaled Error (RMSSE) between predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Defaults to `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean squared error.

    """

    if len(prediction.shape) == 4: # (Bs, L, N, 1) else (Bs, N, 1)
        prediction = torch.mean(prediction, dim=1)
        target = torch.mean(target, dim=1)

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device), atol=eps)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to maintain unbiased MSE calculation
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = (prediction - target) ** 2  # Compute squared error
    loss *= mask  # Apply mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    # Calculate the naive forecast error
    naive_forecast = target[seasonal_period:]
    naive_target = target[:-seasonal_period]
    naive_error = (naive_forecast - naive_target) ** 2
    naive_error = naive_error * mask[seasonal_period:]  # Apply the mask to the naive error
    naive_error = torch.nan_to_num(naive_error)  # Replace any NaNs in the naive error with zero
    n = prediction.shape[0]  # Number of samples
    rmsse = loss / (torch.sum(naive_error) / (n - seasonal_period))
    return torch.mean(rmsse)