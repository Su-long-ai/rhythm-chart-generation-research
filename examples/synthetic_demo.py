from __future__ import annotations

import torch

from rhythm_chart_research.critic import MultiHeadPlayabilityCritic
from rhythm_chart_research.oof import build_group_folds
from rhythm_chart_research.synthetic import make_synthetic_batch


def main() -> None:
    torch.manual_seed(7)
    batch = make_synthetic_batch(seed=7)
    critic = MultiHeadPlayabilityCritic()
    critic.eval()
    with torch.no_grad():
        scores = critic(batch.events, batch.mask, batch.context)
    folds = build_group_folds(batch.groups, n_splits=3, seed=7)
    print("heads:", ", ".join(scores))
    print("score shapes:", {name: tuple(value.shape) for name, value in scores.items()})
    print("held groups:", [fold.held_groups for fold in folds])


if __name__ == "__main__":
    main()
