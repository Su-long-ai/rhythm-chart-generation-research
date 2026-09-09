from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class GroupFold:
    fold: int
    train_indices: np.ndarray
    held_indices: np.ndarray
    train_groups: tuple[Hashable, ...]
    held_groups: tuple[Hashable, ...]


def build_group_folds(
    groups: Sequence[Hashable],
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> list[GroupFold]:
    """Build deterministic group-disjoint folds.

    A group is assigned to exactly one held fold.  This is useful for music or
    creator identities where row-level random splitting would leak identity.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    values = np.asarray(list(groups), dtype=object)
    unique = np.asarray(list(dict.fromkeys(values.tolist())), dtype=object)
    if len(unique) < n_splits:
        raise ValueError("number of unique groups must be >= n_splits")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    buckets = [tuple(x.tolist()) for x in np.array_split(shuffled, n_splits)]

    folds: list[GroupFold] = []
    all_indices = np.arange(len(values), dtype=np.int64)
    for fold, held_groups in enumerate(buckets):
        held_set = set(held_groups)
        held_mask = np.asarray([g in held_set for g in values], dtype=bool)
        train_indices = all_indices[~held_mask]
        held_indices = all_indices[held_mask]
        train_groups = tuple(g for g in unique.tolist() if g not in held_set)
        result = GroupFold(fold, train_indices, held_indices, train_groups, held_groups)
        assert_no_group_leakage(result)
        folds.append(result)
    return folds


def assert_no_group_leakage(fold: GroupFold) -> None:
    overlap = set(fold.train_groups) & set(fold.held_groups)
    if overlap:
        raise AssertionError(f"group leakage in fold {fold.fold}: {sorted(map(str, overlap))}")


def fit_standardizer_train_only(x: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean/std using training rows only; never inspect held-fold statistics."""

    train = np.asarray(x, dtype=np.float32)[train_indices]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) - mean) / std
