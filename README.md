# Rhythm Chart Generation Research

**ML research portfolio project:** a sanitized, runnable showcase for structured 4K rhythm-chart modeling with explicit legality constraints, causal sequence features, multi-head diagnostics, and leakage-aware evaluation.

**Recruiter quick scan**
- structured sequence / typed-event representation
- deterministic legality state machine separated from learned quality signals
- causal lane-recency residual prior
- sequence/context multi-head playability critic
- identity-grouped OOF evaluation with train-only transforms
- reproducible synthetic demo and offline tests
- **Verification:** 7/7 public tests pass in local verification; `ruff check src tests` passes

This public repository intentionally exposes reusable research ideas without publishing private datasets, copyrighted media, replay corpora, checkpoints, or third-party binaries.

## What is included

- deterministic typed-event legality checks with persistent held-lane state;
- a causal learned lane-recency residual prior;
- a sequence/context **multi-head playability critic** that keeps legality, musical timing, gesture, workload and preference-adjacent signals separate;
- deterministic identity-grouped K-fold / out-of-fold utilities with train-only standardization;
- synthetic data and offline tests so the public repository is reproducible without external assets.

## Architecture

```text
synthetic / user-provided event sequence
              |
              +--> deterministic legality state machine
              |
              +--> causal motor context (lane recency)
              |           |
              |           +--> residual press-logit prior
              |
              +--> event features ------> bidirectional sequence encoder --+
              |                                                            |
context / difficulty / section features --> context encoder ---------------+--> fusion
                                                                           |
                                                                           +--> diagnostic heads

identity groups --> deterministic grouped folds --> train-only transforms --> OOF evaluation
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/synthetic_demo.py
```

All tests are offline.  No network, microphone, game installation, beatmap corpus or GPU is required.

## Research stance

The private research workspace treated automatic metrics as diagnostics rather than human-quality truth.  This public extraction preserves that boundary:

1. **legality is deterministic** and is not learned as a replacement for hard checks;
2. **diagnostic targets stay multi-head** instead of being silently collapsed into one reward;
3. **identity/song-like groups stay fold-disjoint** to reduce leakage;
4. scalers and transforms are fit on training rows only;
5. negative and inconclusive experiments should be retained rather than rewritten as successes;
6. human preference claims require actual blinded human evaluation.

See [docs/RESEARCH_METHOD.md](docs/RESEARCH_METHOD.md).

## Public/private boundary

The original research workspace is intentionally private because it contains experiment-specific paths, local corpora, evaluation tooling, and integrations whose licenses/privacy constraints differ.  This repository contains only generic code and synthetic examples.  See [docs/PRIVATE_TO_PUBLIC_BOUNDARY.md](docs/PRIVATE_TO_PUBLIC_BOUNDARY.md).

## Limitations

This showcase does not ship an audio encoder, production generator, real chart dataset, trained checkpoint or human-evaluation corpus.  The synthetic example demonstrates interfaces and invariants, not chart quality.

## License

MIT for the code in this repository.  No license is granted to datasets, media, models or third-party tools that are not included here.
