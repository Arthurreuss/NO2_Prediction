import torch

from src.utils.scale import inverse_target


def evaluate_persistence(
    loader,
    model,
    horizon: int,
    *,
    scaler=None,
    numeric_cols=None,
    target_col: str = None,
):
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            y_pred = model.predict_batch(xb, horizon)  # [B, H]

            # Ensure yb is [B, H]
            if yb.ndim == 3 and yb.shape[-1] == 1:
                yb = yb[..., 0]

            y_true_all.append(yb.detach().cpu())
            y_pred_all.append(y_pred.detach().cpu())

    y_true = torch.cat(y_true_all, dim=0)  # [N, H]
    y_pred = torch.cat(y_pred_all, dim=0)  # [N, H]

    if scaler is not None:
        if numeric_cols is None or target_col is None:
            raise ValueError(
                "If scaler is provided, numeric_cols and target_col must be provided."
            )
        y_true = inverse_target(scaler, y_true, numeric_cols, target_col)
        y_pred = inverse_target(scaler, y_pred, numeric_cols, target_col)

    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

    denom = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    smape = (
        torch.where(
            denom == 0, torch.zeros_like(denom), torch.abs(y_pred - y_true) / denom
        ).mean()
        * 100.0
    ).item()

    return {"rmse": rmse, "smape": smape}
