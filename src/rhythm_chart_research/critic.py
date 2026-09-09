from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

PLAYABILITY_HEADS = (
    "hard_validity",
    "timing_music",
    "local_gesture",
    "hold_state",
    "difficulty_continuity",
    "section_form",
    "style_identity",
    "workload_accumulation",
    "challenge_budget",
    "player_evidence",
)


class MultiHeadPlayabilityCritic(nn.Module):
    """Sequence + context encoder with interpretable diagnostic heads.

    The model is intentionally a diagnostic critic, not a single scalar reward.
    Keeping heads separate avoids pretending that legality, musical alignment,
    workload and preference are interchangeable targets.
    """

    def __init__(
        self,
        event_dim: int = 26,
        context_dim: int = 24,
        hidden: int = 96,
        heads: tuple[str, ...] = PLAYABILITY_HEADS,
    ) -> None:
        super().__init__()
        self.event_in = nn.Sequential(nn.Linear(event_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.sequence = nn.GRU(
            hidden,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 5, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.heads = nn.ModuleDict(
            {name: nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1)) for name in heads}
        )

    def forward(self, events: Tensor, mask: Tensor, context: Tensor) -> Mapping[str, Tensor]:
        if events.ndim != 3 or mask.ndim != 2 or context.ndim != 2:
            raise ValueError("expected events[B,T,D], mask[B,T], context[B,C]")
        if events.shape[:2] != mask.shape:
            raise ValueError("events/mask batch-time dimensions must match")
        encoded, _ = self.sequence(self.event_in(events))
        weight = mask.unsqueeze(-1).to(encoded.dtype)
        denom = weight.sum(dim=1).clamp_min(1.0)
        mean = (encoded * weight).sum(dim=1) / denom
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        fused = self.fusion(torch.cat((mean, maximum, self.context(context)), dim=-1))
        return {name: layer(fused).squeeze(-1) for name, layer in self.heads.items()}
