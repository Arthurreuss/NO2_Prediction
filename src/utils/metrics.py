import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute Root Mean Squared Error (RMSE).

    RMSE measures the average magnitude of the prediction error and is
    expressed in the same units as the target variable.

    Args:
        y_true: Ground-truth values as a tensor.
        y_pred: Predicted values as a tensor with the same shape as `y_true`.

    Returns:
        The RMSE value as a Python float.
    """
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def smape(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is a scale-independent error metric expressed as a percentage.
    It is symmetric with respect to over- and under-predictions and is
    bounded between 0 and 200.

    Args:
        y_true: Ground-truth values as a tensor.
        y_pred: Predicted values as a tensor with the same shape as `y_true`.
        eps: Small constant to avoid division by zero.

    Returns:
        The SMAPE value (in percent) as a Python float.
    """
    num = torch.abs(y_true - y_pred)
    denom = torch.clamp(torch.abs(y_true) + torch.abs(y_pred), min=eps)
    return (200.0 * torch.mean(num / denom)).item()
