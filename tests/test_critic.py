import torch

from rhythm_chart_research.critic import MultiHeadPlayabilityCritic, PLAYABILITY_HEADS
from rhythm_chart_research.synthetic import make_synthetic_batch


def test_all_heads_return_one_score_per_sample() -> None:
    torch.manual_seed(1)
    batch = make_synthetic_batch(batch_size=6, max_events=12)
    model = MultiHeadPlayabilityCritic(hidden=16)
    model.eval()
    with torch.no_grad():
        outputs = model(batch.events, batch.mask, batch.context)
    assert tuple(outputs) == PLAYABILITY_HEADS
    assert all(value.shape == (6,) for value in outputs.values())
    assert all(torch.isfinite(value).all() for value in outputs.values())
