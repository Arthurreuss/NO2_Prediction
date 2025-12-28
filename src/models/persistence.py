import torch


class PersistenceModel:
    """Persistence baseline model for time-series forecasting.

    This baseline predicts all future values as the last observed value
    of the target feature in the input window. It assumes that the target
    variable (e.g., nitrogen dioxide) is included among the input features.
    """

    def __init__(self, target_feature_index: int) -> None:
        """Initialize the persistence model.

        Args:
            target_feature_index: Index of the target feature (e.g.,
                "nitrogen_dioxide") within the feature dimension of the
                input tensor provided by the DataLoader.
        """
        self.target_feature_index = target_feature_index

    def predict_batch(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """Predict a batch of future values using persistence.

        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features].
            horizon: Forecast horizon (number of future time steps to predict).

        Returns:
            A tensor of shape [batch_size, horizon] where each future step
            is equal to the last observed target value in the input window.
        """
        last_step = x[:, -1, self.target_feature_index]  # [B]
        y_hat = last_step.unsqueeze(1).repeat(1, horizon)  # [B, H]
        return y_hat
