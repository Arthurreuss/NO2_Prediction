from typing import Dict, List

import torch
import torch.nn as nn


class MultiGRUForecast(nn.Module):
    """
    Hierarchical GRU:
      - shared GRU backbone over all features
      - separate MLP branch per pollutant (subtask)
    Returns tensor of shape [B, H, T] with T = n_targets.
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
    ):
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
        self.branches = nn.ModuleDict(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F]
        returns: y_hat [B, H, T] where T = len(target_cols)
        """
        out, _ = self.gru(x)  # [B, L, Hshared]
        h_last = out[:, -1, :]  # [B, Hshared]
        h_last = self.shared_dropout(h_last)

        preds = []
        for name in self.target_cols:
            y_t = self.branches[name](h_last)  # [B, H]
            preds.append(y_t.unsqueeze(-1))  # [B, H, 1]

        y_hat = torch.cat(preds, dim=-1)  # [B, H, T]
        return y_hat
