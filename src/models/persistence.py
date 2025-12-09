import torch


class PersistenceModel:
    """
    Baseline: predicts future NO2 as the last observed NO2 in the input window.
    Assumes the target NO2 is one of the feature columns.
    """

    def __init__(self, target_feature_index: int):
        """
        target_feature_index: index of 'nitrogen_dioxide' inside the feature_cols list
        used by the DataLoader.
        """
        self.target_feature_index = target_feature_index

    def predict_batch(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        x: [B, L, F]
        returns y_hat: [B, H]
        """
        last_step = x[:, -1, self.target_feature_index]  # [B]
        y_hat = last_step.unsqueeze(1).repeat(1, horizon)  # [B, H]
        return y_hat
