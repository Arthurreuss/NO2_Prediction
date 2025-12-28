from typing import List

import torch
import torch.nn as nn


class HierarchicalGRUForecast(nn.Module):
    """Hierarchical GRU for multi-target forecasting with fused short/long context.

    Architecture:
      - Short-term GRU encodes fine-grained inputs and produces:
          - `short_out`: per-step hidden states [B, L, H_short]
          - `short_hn`: final hidden state(s) [num_layers_short, B, H_short]
      - Downsampling mean-pools `short_out` every `downsample_factor` steps to produce
        a coarse sequence [B, L_coarse, H_short]
      - Long-term GRU encodes the coarse sequence and produces:
          - `long_hn`: final hidden state(s) [num_layers_long, B, H_long]
      - Feature fusion concatenates the last-layer final states from long- and short-term
        encoders to form a combined representation [B, H_long + H_short]
      - Decoder maps the combined representation to horizon x targets

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
        """Initialize the hierarchical GRU model with fused decoder input.

        Args:
            input_size: Number of input features per time step.
            target_cols: Names of target variables to predict.
            horizon: Forecast horizon (number of future steps predicted).
            downsample_factor: Number of fine-grained steps to aggregate into one
                coarse step.
            short_hidden_size: Hidden size of the short-term GRU.
            long_hidden_size: Hidden size of the long-term GRU.
            num_layers_short: Number of layers in the short-term GRU.
            num_layers_long: Number of layers in the long-term GRU.
            dropout: Dropout probability applied within GRUs (when layers > 1)
                and within the decoder.
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
        decoder_input_size = long_hidden_size + short_hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, long_hidden_size),
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
            h_seq: Hidden-state sequence of shape [B, L, H_short].

        Returns:
            A coarse hidden-state sequence of shape [B, L_coarse, H_short].
        """
        B, L, Hs = h_seq.shape
        k = self.downsample_factor

        if L < k:
            return h_seq.mean(dim=1, keepdim=True)

        L_trim = (L // k) * k
        h_trim = h_seq[:, :L_trim, :]

        L_coarse = L_trim // k
        h_trim = h_trim.view(B, L_coarse, k, Hs)
        h_coarse = h_trim.mean(dim=2)

        return h_coarse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass of the model.

        Args:
            x: Input tensor of normalized features with shape [B, L, F].

        Returns:
            A tensor of shape [B, horizon, n_targets] containing the forecasts.
        """
        short_out, short_hn = self.short_gru(x)
        coarse_seq = self._downsample_hidden(short_out)
        _, long_hn = self.long_gru(coarse_seq)

        final_short = short_hn[-1]
        final_long = long_hn[-1]

        combined_state = torch.cat([final_long, final_short], dim=1)

        dec = self.decoder(combined_state)
        y_hat = dec.view(-1, self.horizon, self.n_targets)

        return y_hat
