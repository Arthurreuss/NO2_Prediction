from typing import Dict, List

import torch
import torch.nn as nn


class MultiGRUForecast(nn.Module):
    """Multi-target GRU forecaster with a shared backbone and per-target branches.

    Architecture:
      - Shared GRU backbone over all input features to produce a shared representation.
      - Separate MLP branch per target (subtask) to predict a full horizon.

    Output shape:
        [B, H, T] where:
          - B is batch size
          - H is forecast horizon
          - T is the number of targets (len(target_cols))
    """

    def __init__(
        self,
        input_size: int,
        target_cols: List[str],
        horizon: int = 24,
        shared_hidden_size: int = 32,
        branch_hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the multi-target GRU forecasting model.

        Args:
            input_size: Number of input features per time step.
            target_cols: Names of target variables to predict.
            horizon: Forecast horizon (number of future steps predicted).
            shared_hidden_size: Hidden size of the shared GRU backbone.
            branch_hidden_size: Hidden size of each target-specific MLP branch.
            num_layers: Number of stacked GRU layers in the shared backbone.
            dropout: Dropout probability applied after the shared GRU output and
                within each target-specific branch. GRU-internal dropout is only
                active when `num_layers > 1`.
        """
        super().__init__()
        self.target_cols = target_cols
        self.horizon = horizon

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=shared_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.shared_dropout = nn.Dropout(dropout)

        branches: Dict[str, nn.Module] = {}
        for name in target_cols:
            branches[name] = nn.Sequential(
                nn.Linear(shared_hidden_size, branch_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(branch_hidden_size, horizon),
            )
        self.branches: nn.ModuleDict = nn.ModuleDict(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass of the model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size].

        Returns:
            A tensor of shape [batch_size, horizon, n_targets] containing the
            forecast for each target over the full horizon.
        """
        out, _ = self.gru(x)  # [B, L, Hshared]
        h_last = out[:, -1, :]  # [B, Hshared]
        h_last = self.shared_dropout(h_last)

        preds: List[torch.Tensor] = []
        for name in self.target_cols:
            y_t = self.branches[name](h_last)  # [B, H]
            preds.append(y_t.unsqueeze(-1))  # [B, H, 1]

        y_hat = torch.cat(preds, dim=-1)  # [B, H, T]
        return y_hat
