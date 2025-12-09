import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def smape(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> float:
    num = torch.abs(y_true - y_pred)
    denom = torch.clamp(torch.abs(y_true) + torch.abs(y_pred), min=eps)
    return (200.0 * torch.mean(num / denom)).item()
