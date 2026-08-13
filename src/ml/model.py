"""Day-range predictor models:

- DayRangeGRU (original primary): GRU over the daily sequence + flattened
  MLP over the intraday bars, concatenated into a small MLP head.
- DayRangeMLP (fallback): same intraday branch, daily branch flattened
  instead of recurrent.
- DayRangeTransformer: now that daily and intraday bars share an identical
  5-feature schema (src/ml/features.py), this concatenates them into ONE
  flat sequence and encodes it with a standard Transformer encoder (a
  learned segment embedding distinguishes daily vs intraday positions,
  since they're different time granularities despite the shared schema),
  pooling a prepended CLS token's output through the head.

All three take (daily: [B, daily_lookback, 5], intraday: [B, intraday_bars,
5]) and return [B, 2] = [pred_high_ret, pred_low_ret]. Every constructor
accepts **_ignored so the training script can pass one superset of
hyperparameters regardless of which encoder is selected.
"""

import math

import torch
from torch import nn

DAILY_FEATURES = 5
INTRADAY_FEATURES = 5


def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Classic fixed (non-learned) sin/cos positional encoding — unlike a
    learned nn.Embedding, this carries useful positional signal from step
    zero rather than starting as random noise the model has to learn,
    which matters more on a dataset this small.
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class DayRangeGRU(nn.Module):
    def __init__(
        self,
        daily_lookback: int,
        intraday_bars: int,
        daily_hidden: int = 32,
        intraday_hidden: int = 16,
        head_hidden: int = 32,
        dropout: float = 0.2,
        **_ignored,
    ):
        super().__init__()
        self.daily_gru = nn.GRU(
            DAILY_FEATURES, daily_hidden, num_layers=1, batch_first=True
        )
        self.intraday_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intraday_bars * INTRADAY_FEATURES, intraday_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(daily_hidden + intraday_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, daily: torch.Tensor, intraday: torch.Tensor) -> torch.Tensor:
        _, h_n = self.daily_gru(daily)  # h_n: [1, B, daily_hidden]
        daily_repr = h_n.squeeze(0)
        intraday_repr = self.intraday_mlp(intraday)
        return self.head(torch.cat([daily_repr, intraday_repr], dim=-1))


class DayRangeMLP(nn.Module):
    """Fallback encoder: replaces the GRU with a second flattened MLP
    branch over the daily window, treating the 14 days as an unordered
    flat feature vector rather than a sequence.
    """

    def __init__(
        self,
        daily_lookback: int,
        intraday_bars: int,
        daily_hidden: int = 32,
        intraday_hidden: int = 16,
        head_hidden: int = 32,
        dropout: float = 0.2,
        **_ignored,
    ):
        super().__init__()
        self.daily_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(daily_lookback * DAILY_FEATURES, daily_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.intraday_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intraday_bars * INTRADAY_FEATURES, intraday_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(daily_hidden + intraday_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, daily: torch.Tensor, intraday: torch.Tensor) -> torch.Tensor:
        daily_repr = self.daily_mlp(daily)
        intraday_repr = self.intraday_mlp(intraday)
        return self.head(torch.cat([daily_repr, intraday_repr], dim=-1))


class DayRangeTransformer(nn.Module):
    def __init__(
        self,
        daily_lookback: int,
        intraday_bars: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        head_hidden: int = 32,
        dropout: float = 0.2,
        **_ignored,
    ):
        super().__init__()
        seq_len = daily_lookback + intraday_bars
        self.input_proj = nn.Linear(DAILY_FEATURES, d_model)
        self.segment_emb = nn.Embedding(2, d_model)  # 0=daily bar, 1=intraday bar
        self.register_buffer(
            "pos_encoding", _sinusoidal_positional_encoding(seq_len + 1, d_model)
        )  # +1 for the CLS token; fixed, not learned -- see the helper's docstring
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN -- PyTorch's Post-LN default is prone to
            # exactly the "stuck in an early plateau" symptom seen without this
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )
        segment_ids = torch.cat(
            [
                torch.zeros(daily_lookback, dtype=torch.long),
                torch.ones(intraday_bars, dtype=torch.long),
            ]
        )
        self.register_buffer("segment_ids", segment_ids)

    def forward(self, daily: torch.Tensor, intraday: torch.Tensor) -> torch.Tensor:
        batch = daily.shape[0]
        seq = torch.cat([daily, intraday], dim=1)  # [B, seq_len, 5]
        x = self.input_proj(seq) + self.segment_emb(self.segment_ids).unsqueeze(0)
        cls = self.cls_token.expand(batch, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)  # [B, 1+seq_len, d_model]
        x = x + self.pos_encoding.unsqueeze(0)
        encoded = self.encoder(x)  # [B, 1+seq_len, d_model]
        cls_out = encoded[:, 0, :]  # pooled CLS representation
        return self.head(cls_out)


def build_model(encoder: str, **kwargs) -> nn.Module:
    if encoder == "gru":
        return DayRangeGRU(**kwargs)
    if encoder == "mlp":
        return DayRangeMLP(**kwargs)
    if encoder == "transformer":
        return DayRangeTransformer(**kwargs)
    raise ValueError(
        f"Unknown encoder: {encoder!r} (expected 'gru', 'mlp', or 'transformer')"
    )
