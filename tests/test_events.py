from rhythm_chart_research.events import Event, EventKind, validate_events


def test_valid_event_sequence() -> None:
    report = validate_events([
        Event(0.0, 0),
        Event(0.5, 1, EventKind.HOLD, 1.0),
        Event(1.5, 1),
    ])
    assert report.valid
    assert report.error_count == 0


def test_overlap_and_invalid_hold_are_separate_errors() -> None:
    report = validate_events([
        Event(0.0, 2, EventKind.HOLD, 2.0),
        Event(1.0, 2),
        Event(3.0, 3, EventKind.HOLD, 0.0),
    ])
    assert not report.valid
    assert report.held_lane_overlap == 1
    assert report.nonpositive_hold == 1
