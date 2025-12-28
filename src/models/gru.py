import torch
import torch.nn as nn


class GRUForecast(nn.Module):
    """GRU-based neural network for multi-step time-series forecasting.

    The model encodes an input sequence using a GRU and predicts a fixed
    forecast horizon from the final hidden state via a linear layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        horizon: int = 24,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the GRU forecasting model.

        Args:
            input_size: Number of input features per time step.
            hidden_size: Number of hidden units in the GRU.
            num_layers: Number of stacked GRU layers.
            horizon: Forecast horizon (number of future time steps predicted).
            dropout: Dropout probability applied after the GRU. Note that
                GRU-internal dropout is only active when `num_layers > 1`.
        """
        super().__init__()
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(0.0 if num_layers == 1 else dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass of the model.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size].

        Returns:
            A tensor of shape [batch_size, horizon] containing the model's
            forecast for each sample in the batch.
        """
        out, _ = self.gru(x)  # out: [B, L, Hhid]
        last_hidden = out[:, -1, :]  # [B, Hhid]
        last_hidden = self.dropout(last_hidden)  # G: dropout
        y_hat = self.fc(last_hidden)  # [B, H]
        return y_hat
