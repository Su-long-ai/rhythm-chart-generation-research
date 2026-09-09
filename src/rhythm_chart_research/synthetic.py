from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SyntheticBatch:
    events: Tensor
    mask: Tensor
    context: Tensor
    groups: np.ndarray


def make_synthetic_batch(
    *,
    batch_size: int = 12,
    max_events: int = 32,
    event_dim: int = 26,
    context_dim: int = 24,
    seed: int = 7,
) -> SyntheticBatch:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(low=max(2, max_events // 3), high=max_events + 1, size=batch_size)
    events = torch.zeros(batch_size, max_events, event_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_events, dtype=torch.bool)
    for i, length in enumerate(lengths.tolist()):
        events[i, :length] = torch.from_numpy(rng.normal(size=(length, event_dim)).astype(np.float32))
        mask[i, :length] = True
    context = torch.from_numpy(rng.normal(size=(batch_size, context_dim)).astype(np.float32))
    groups = np.asarray([f"identity-{i // 2}" for i in range(batch_size)], dtype=object)
    return SyntheticBatch(events, mask, context, groups)
