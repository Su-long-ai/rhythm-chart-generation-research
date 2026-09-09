import torch

from rhythm_chart_research.recency import LaneRecencyPrior, lane_recency_features


def test_recency_is_causal_and_shaped() -> None:
    features = lane_recency_features([0.0, 0.5, 1.0], [0, 1, 0])
    assert features.shape == (3, 6)
    assert features[2, 0] < features[2, 2]


def test_zero_initialized_prior_is_identity_adjustment() -> None:
    prior = LaneRecencyPrior()
    features = torch.ones(5, 6)
    logits = torch.randn(5, 4)
    adjusted, correction = prior.adjust(logits, features)
    assert torch.allclose(correction, torch.zeros_like(correction))
    assert torch.allclose(adjusted, logits)
