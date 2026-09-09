import numpy as np

from rhythm_chart_research.oof import build_group_folds, fit_standardizer_train_only


def test_group_folds_are_disjoint_and_deterministic() -> None:
    groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    a = build_group_folds(groups, n_splits=5, seed=11)
    b = build_group_folds(groups, n_splits=5, seed=11)
    assert [f.held_groups for f in a] == [f.held_groups for f in b]
    assert set().union(*(set(f.held_groups) for f in a)) == set(groups)
    for fold in a:
        assert not (set(fold.train_groups) & set(fold.held_groups))


def test_standardizer_uses_train_rows_only() -> None:
    x = np.asarray([[0.0], [2.0], [1000.0]], dtype=np.float32)
    mean, std = fit_standardizer_train_only(x, np.asarray([0, 1]))
    assert np.allclose(mean, [1.0])
    assert np.allclose(std, [1.0])
