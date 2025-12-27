from typing import List

import torch
import torch.nn as nn


class HierarchicalGRUForecast(nn.Module):
    """Hierarchical GRU for multi-target air-quality forecasting.

    Architecture:
      - Level 1 (short-term): GRU over hourly inputs [B, L, F] -> [B, L, H_short]
      - Temporal downsampling: groups every `downsample_factor` steps and mean-pools
        to produce a coarse sequence [B, L_coarse, H_short]
      - Level 2 (long-term): GRU over coarse sequence -> hidden state [B, H_long]
      - Decoder: maps final long-term hidden state to horizon x targets

    Output shape:
        [B, horizon, T] where T = len(target_cols)
    """

    def __init__(
        self,
        input_size: int,
        target_cols: List[str],
        horizon: int,
        downsample_factor: int = 24,
        short_hidden_size: int = 64,
        long_hidden_size: int = 32,
        num_layers_short: int = 1,
        num_layers_long: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the hierarchical GRU model.

        Args:
            input_size: Number of input features per time step.
            target_cols: Names of target variables to predict.
            horizon: Forecast horizon (number of future steps predicted).
            downsample_factor: Number of fine-grained steps to aggregate into one
                coarse step (e.g., 24 for daily aggregation from hourly inputs).
            short_hidden_size: Hidden size of the short-term GRU.
            long_hidden_size: Hidden size of the long-term GRU.
            num_layers_short: Number of layers in the short-term GRU.
            num_layers_long: Number of layers in the long-term GRU.
            dropout: Dropout probability used in GRUs (when layers > 1) and decoder.
        """
        super().__init__()
        self.input_size = input_size
        self.target_cols = target_cols
        self.n_targets = len(target_cols)
        self.horizon = horizon
        self.downsample_factor = downsample_factor

        self.short_gru = nn.GRU(
            input_size=input_size,
            hidden_size=short_hidden_size,
            num_layers=num_layers_short,
            batch_first=True,
            dropout=dropout if num_layers_short > 1 else 0.0,
        )

        self.long_gru = nn.GRU(
            input_size=short_hidden_size,
            hidden_size=long_hidden_size,
            num_layers=num_layers_long,
            batch_first=True,
            dropout=dropout if num_layers_long > 1 else 0.0,
        )

        self.decoder = nn.Sequential(
            nn.Linear(long_hidden_size, long_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(long_hidden_size, self.horizon * self.n_targets),
        )

    def _downsample_hidden(self, h_seq: torch.Tensor) -> torch.Tensor:
        """Downsample a hidden-state sequence via mean pooling over fixed groups.

        The sequence is grouped into contiguous chunks of length
        `self.downsample_factor` and averaged within each chunk. If the
        sequence length is shorter than the downsample factor, the mean over
        the full sequence is returned as a single coarse step.

        Args:
            h_seq: Hidden-state sequence of shape [batch_size, seq_len, H_short].

        Returns:
            A coarse hidden-state sequence of shape [batch_size, seq_len_coarse, H_short].
        """
        B, L, Hs = h_seq.shape
        k = self.downsample_factor

        if L < k:
            return h_seq.mean(dim=1, keepdim=True)  # [B, 1, Hs]

        L_trim = (L // k) * k
        h_trim = h_seq[:, :L_trim, :]  # [B, L_trim, Hs]

        L_coarse = L_trim // k
        h_trim = h_trim.view(B, L_coarse, k, Hs)
        h_coarse = h_trim.mean(dim=2)  # [B, L_coarse, Hs]

        return h_coarse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass of the model.

        Args:
            x: Input tensor of normalized features with shape
                [batch_size, seq_len, input_size].

        Returns:
            A tensor of shape [batch_size, horizon, n_targets] containing the
            model's forecasts for each target over the forecast horizon.
        """
        short_out, _ = self.short_gru(x)  # [B, L, H_short]

        coarse_seq = self._downsample_hidden(short_out)  # [B, L_coarse, H_short]

        long_out, long_hn = self.long_gru(
            coarse_seq
        )  # long_hn: [num_layers_long, B, H_long]

        # long_hn[-1]: [B, H_long]
        h_final = long_hn[-1]  # [B, H_long]

        dec = self.decoder(h_final)  # [B, horizon * T]
        y_hat = dec.view(-1, self.horizon, self.n_targets)  # [B, H, T]

        return y_hat
