# Research Method

The goal of this public repository is to make the **experimental discipline** inspectable without publishing private corpora or third-party assets.

## 1. Separate hard invariants from learned judgments

State-machine legality (lane range, positive hold duration, no held-lane overlap, duplicate starts) is checked deterministically.  A learned critic can diagnose richer properties, but it should not be the sole authority for a rule that can be checked exactly.

## 2. Prefer interpretable diagnostic heads

`MultiHeadPlayabilityCritic` exposes separate heads for validity, timing/music, local gesture, hold state, difficulty continuity, section form, style identity, accumulated workload, challenge budget and player evidence.  These names are hypotheses/targets, not claims that the public synthetic model has learned human truth.

## 3. Group-disjoint evaluation

Rows from the same identity or source should not be randomly split across train and held folds. `build_group_folds` assigns each group to exactly one held fold and asserts disjointness.  Data-derived transforms are fit with `fit_standardizer_train_only` on train indices only.

## 4. Reproducibility

Public examples use fixed seeds.  A serious experiment should also freeze the data manifest, feature schema, model configuration, evaluation protocol and artifact hashes before opening held-out metrics.

## 5. Falsification and negative results

A failed hypothesis is evidence.  Keep negative/inconclusive runs with their protocol and failure reason instead of deleting them or changing the target after seeing validation results.

## 6. Human evaluation boundary

Automatic legality, replay-derived signals, heuristics and synthetic corruptions cannot substitute for blinded actual-play preference judgments.  Any future human-quality claim should state annotator protocol, blinding, sample size, exclusions and uncertainty.
