import torch
import torch.nn as nn


class GRUForecast(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        horizon: int = 24,
        dropout: float = 0.2,
    ):
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
        """
        x: [B, L, F]
        returns: [B, H]
        """
        out, _ = self.gru(x)  # out: [B, L, Hhid]
        last_hidden = out[:, -1, :]  # [B, Hhid]
        last_hidden = self.dropout(last_hidden)  # G: dropout
        y_hat = self.fc(last_hidden)  # [B, H]
        return y_hat
