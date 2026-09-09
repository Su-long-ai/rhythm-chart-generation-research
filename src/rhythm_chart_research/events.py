from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable


class EventKind(str, Enum):
    TAP = "tap"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Event:
    """One 4K note event in beat coordinates.

    Real beatmap parsing is intentionally outside this public showcase.  The
    representation is generic and works with synthetic or user-provided data.
    """

    beat: float
    lane: int
    kind: EventKind = EventKind.TAP
    duration_beats: float = 0.0


@dataclass(frozen=True, slots=True)
class LegalityReport:
    valid: bool
    non_finite: int = 0
    lane_out_of_range: int = 0
    tap_with_duration: int = 0
    nonpositive_hold: int = 0
    held_lane_overlap: int = 0
    duplicate_start: int = 0

    @property
    def error_count(self) -> int:
        return (
            self.non_finite
            + self.lane_out_of_range
            + self.tap_with_duration
            + self.nonpositive_hold
            + self.held_lane_overlap
            + self.duplicate_start
        )


def validate_events(events: Iterable[Event], lanes: int = 4) -> LegalityReport:
    """Check deterministic state-machine legality without learning a quality score."""

    counts = {
        "non_finite": 0,
        "lane_out_of_range": 0,
        "tap_with_duration": 0,
        "nonpositive_hold": 0,
        "held_lane_overlap": 0,
        "duplicate_start": 0,
    }
    held_until: dict[int, float] = {}
    seen_starts: set[tuple[float, int]] = set()

    for event in sorted(events, key=lambda e: (e.beat, e.lane)):
        if not math.isfinite(event.beat) or not math.isfinite(event.duration_beats):
            counts["non_finite"] += 1
            continue
        if not 0 <= event.lane < lanes:
            counts["lane_out_of_range"] += 1
            continue
        key = (event.beat, event.lane)
        if key in seen_starts:
            counts["duplicate_start"] += 1
        seen_starts.add(key)

        if event.kind is EventKind.TAP:
            if abs(event.duration_beats) > 1e-9:
                counts["tap_with_duration"] += 1
        elif event.kind is EventKind.HOLD:
            if event.duration_beats <= 0:
                counts["nonpositive_hold"] += 1

        if held_until.get(event.lane, -math.inf) > event.beat + 1e-9:
            counts["held_lane_overlap"] += 1
        if event.kind is EventKind.HOLD and event.duration_beats > 0:
            held_until[event.lane] = event.beat + event.duration_beats

    total = sum(counts.values())
    return LegalityReport(valid=total == 0, **counts)
