from typing import Callable

import torch
from torch.utils.data import DataLoader

from .metrics import rmse, smape


@torch.no_grad()
def evaluate_persistence(
    dataloader: DataLoader,
    model,
    horizon: int,
    device: str,
) -> dict:
    y_true_all = []
    y_pred_all = []

    for x, y in dataloader:
        x = x.to(device)
        y_hat = model.predict_batch(x, horizon=horizon)  # [B, H]

        y_true_all.append(y)  # still [B, H]
        y_pred_all.append(y_hat.cpu())

    y_true = torch.cat(y_true_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0)

    return {
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
