from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn


def lane_recency_features(
    event_times: Sequence[float],
    event_lanes: Sequence[int],
    *,
    lanes: int = 4,
    clip_gap: float = 8.0,
) -> Tensor:
    """Return per-event motor-context features using only past lane usage.

    Output columns are ``log1p(gap_to_each_lane)`` plus two symmetric hand-load
    summaries.  No future event is read, which makes the feature suitable for
    causal generation experiments.
    """

    if len(event_times) != len(event_lanes):
        raise ValueError("event_times and event_lanes must have equal length")
    last = [-math.inf] * lanes
    rows: list[list[float]] = []
    for time, lane in zip(event_times, event_lanes):
        if not 0 <= lane < lanes:
            raise ValueError(f"lane out of range: {lane}")
        gaps = [clip_gap if not math.isfinite(t) else min(max(time - t, 0.0), clip_gap) for t in last]
        left = 0.5 * (gaps[0] + gaps[1]) if lanes >= 2 else gaps[0]
        right = 0.5 * (gaps[2] + gaps[3]) if lanes >= 4 else left
        rows.append([*(math.log1p(g) for g in gaps), math.log1p(left), math.log1p(right)])
        last[lane] = time
    return torch.tensor(rows, dtype=torch.float32)


class LaneRecencyPrior(nn.Module):
    """Small residual prior that adjusts press logits from causal recency context."""

    def __init__(self, feature_dim: int = 6, hidden: int = 32, output_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)

    def adjust(self, press_logits: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        correction = self(features)
        if press_logits.shape != correction.shape:
            raise ValueError(f"shape mismatch: logits={press_logits.shape}, correction={correction.shape}")
        return press_logits + correction, correction
