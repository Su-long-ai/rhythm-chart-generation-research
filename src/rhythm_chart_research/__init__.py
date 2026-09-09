"""Public, synthetic-data-only components for structured rhythm-chart research."""

from .events import Event, EventKind, LegalityReport, validate_events
from .critic import MultiHeadPlayabilityCritic, PLAYABILITY_HEADS
from .recency import LaneRecencyPrior, lane_recency_features
from .oof import GroupFold, build_group_folds, assert_no_group_leakage

__all__ = [
    "Event",
    "EventKind",
    "LegalityReport",
    "validate_events",
    "MultiHeadPlayabilityCritic",
    "PLAYABILITY_HEADS",
    "LaneRecencyPrior",
    "lane_recency_features",
    "GroupFold",
    "build_group_folds",
    "assert_no_group_leakage",
]
