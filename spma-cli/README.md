# spma-cli

Command-line interface for [spma](https://crates.io/crates/spma) — unsupervised anomaly detection for discrete event sequences.

## Install

```sh
cargo install spma-cli
```

## Quickstart

Input format: one sequence per line, tokens space-separated.

```
TRIP BREAKER_OPEN UNDERVOLTAGE BACKUP_RELAY
TRIP BREAKER_OPEN OVERCURRENT  BACKUP_RELAY
```

```sh
# Train on normal sequences
spma train --corpus normal.txt --output model.json

# Score sequences (exits 1 if any anomaly detected)
spma infer --model model.json --input sequences.txt

# JSON output
spma infer --model model.json --json < sequences.txt

# Extend existing model without cold start
spma retrain --model model.json --corpus new_normal.txt

# Refit thresholds without retraining the grammar
spma recalibrate --model model.json --corpus new_normal.txt

# Inspect the learned grammar
spma grammar --model model.json
spma grammar --model model.json --json --level 0
```

## Corpus validation

Sequences longer than 512 symbols are rejected automatically:

```
Error: corpus validation failed: sequence 3 has 513 symbols (limit: 512)
```

## Per-level anomaly gates

Flag sequences that violate higher-level composition (e.g. correct events in wrong order):

```sh
spma infer --model model.json --level-threshold 0:0.2 --level-threshold 1:0.5
```

## Documentation

- [Architecture](https://github.com/geronimo-iia/spma/blob/main/docs/architecture.md) — scoring objective, beam search, learning loop
- [Scoring](https://github.com/geronimo-iia/spma/blob/main/docs/scoring.md) — E_norm, threshold semantics, per-level calibration
- [Known limitations](https://github.com/geronimo-iia/spma/blob/main/docs/known-limitations.md)

Library crate: [spma](https://crates.io/crates/spma)
