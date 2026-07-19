# spma

Unsupervised anomaly detection for discrete event sequences via MDL-based grammar induction (SP Multiple Alignment).

Learns a hierarchical grammar from normal sequences, then scores new sequences by their encoding cost E. Sequences that compress poorly — high `e_norm` — are anomalous.


## Background

I'm a software engineer, not an ML researcher. This started as a question: is there a symbolic, interpretable approach to anomaly detection that doesn't require neural networks, labeled data, or feature engineering?

That question led me to [J.G. Wolff's SP theory](https://www.cognitionresearch.org) — the idea that intelligence reduces to a single operation: find the best multiple alignment between new input and stored patterns, scored by information compression.

The scoring objective is two-part MDL:

```
T = G + E
```

- **G**: grammar cost — size of what you know
- **E**: encoding cost — cost of explaining new input using the grammar
- High E = the grammar does not explain this input = anomaly

I read the primary sources, found the math sound, found the empirical record thin, and decided to implement the core idea for the problem I actually had: discrete event sequences from industrial systems. This is exploration — I'm not claiming to advance the field, just curious enough to build something and see where it lands.

`spma` is that implementation. Not general AI — the anomaly detection subset: learn a grammar from normal sequences, score new sequences by E.

Three properties transfer from SP theory:

- **No catastrophic forgetting**: grammar is append-only — old patterns are never overwritten
- **Structural transparency**: every inference produces an alignment table showing which patterns matched which events and at what cost — no post-hoc explanation needed
- **Append-only grammar**: patterns are never removed or overwritten — the grammar only grows, which means no forgetting and no retraining from scratch when new normal data arrives

## Install

```toml
[dependencies]
spma = "0.2"
```

Or:

```sh
cargo add spma
```

## Usage

```rust
use spma::Spma;

// Train on normal sequences
let corpus: Vec<Vec<&str>> = vec![
    vec!["TRIP", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
    vec!["TRIP", "BREAKER_OPEN", "OVERCURRENT",  "BACKUP_RELAY"],
    // ... more normal sequences
];
let mut model = Spma::new(10); // beam width = 10
model.train(&corpus);

// Infer a new sequence
let result = model.infer(&["TRIP", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"]);
println!("e_norm={:.3}  anomaly={}", result.e_norm, result.is_anomaly);
println!("{}", result.alignment);
```

`InferResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `e_cost` | `f64` | Encoding cost of unmatched symbols (bits) |
| `e_norm` | `f64` | E normalised by raw sequence cost; 0 = fully covered |
| `is_anomaly` | `bool` | `true` when `e_norm > threshold` (default: any uncovered symbol) |
| `anomaly_percentile` | `f64` | Fraction of training sequences with lower `e_norm` |
| `cd` | `f64` | Compression difference (bits saved by grammar patterns); higher = more structure matched — use alongside `e_norm` as a confidence signal |
| `level_costs` | `Vec<f64>` | Encoding cost per grammar level |
| `level_e_norms` | `Vec<f64>` | Normalised cost per grammar level |
| `alignment` | `Alignment` | Match table — which grammar patterns covered which symbols |

## Persist a model

```rust
use std::io::{BufWriter, BufReader};
use std::fs::File;

// inside a Result-returning function
model.save(BufWriter::new(File::create("model.json")?))?;
let loaded = Spma::load(BufReader::new(File::open("model.json")?))?;
```

## Retrain on new data

```rust
// Extend the grammar with new normal sequences — no cold start.
// Prior patterns and atom frequencies are preserved.
let new_corpus: Vec<Vec<&str>> = vec![
    vec!["TRIP", "BREAKER_OPEN", "OVERHEAT", "BACKUP_RELAY"],
    // ... more sequences
];
model.retrain(&new_corpus);
```

## Recalibrate thresholds

```rust
// Refit e_distribution on a larger normal corpus without retraining the grammar.
// Useful when you train on a small corpus for speed then collect more normal data.
// User-set threshold and level_thresholds are preserved across recalibration.
let new_corpus: Vec<Vec<&str>> = vec![
    vec!["TRIP", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
    // ... more normal sequences
];
model.recalibrate(&new_corpus);
```

## Validate before training

Sequences longer than 512 symbols cannot be processed (bitmask limit). Validate before calling `train`/`retrain`/`recalibrate`:

```rust
use spma::{validate_corpus, validate_sequence};

validate_corpus(&corpus).map_err(|e| format!("corpus error: {e}"))?;
// or for a single sequence:
validate_sequence(&tokens).map_err(|e| format!("sequence error: {e}"))?;
```

The CLI validates automatically and exits with a clear error message. The lib functions are opt-in — callers control when validation runs.

## Examples

Runnable examples covering golden path, save/load, order detection, and threshold tuning — see [examples/README.md](examples/README.md).

## CLI quickstart

```sh
cargo install spma-cli
```

```sh
# Train
spma train --corpus normal.txt --output model.json

# Infer (exits 1 if any sequence is anomalous)
spma infer --model model.json --input sequences.txt
spma infer --model model.json --json < sequences.txt

# Extend existing model with new sequences (no cold start)
spma retrain --model model.json --corpus new_normal.txt

# Refit e_distribution without re-training
spma recalibrate --model model.json --corpus new_normal.txt

# Corpus and sequences are validated automatically — sequences > 512 symbols are rejected:
# Error: corpus validation failed: sequence 3 has 513 symbols (limit: 512)

# Inspect grammar
spma grammar --model model.json
spma grammar --model model.json --json --level 0
```

Input format: one sequence per line, tokens space-separated.

```
TRIP BREAKER_OPEN UNDERVOLTAGE BACKUP_RELAY
TRIP BREAKER_OPEN OVERCURRENT  BACKUP_RELAY
```

### Per-level anomaly gates

Flag sequences that look normal at the atom level but violate higher-level composition — e.g. correct events in wrong order.

```sh
spma infer --model model.json --level-threshold 0:0.2 --level-threshold 1:0.5
```

## How it works

1. **Train**: extracts frequent n-grams, then runs MDL-gated beam search to build a hierarchical grammar of patterns. The MDL gate accepts a new pattern only if it reduces total description length T = G + E — patterns that don't compress are discarded.
2. **Gap patterns**: co-occurring symbols with variable-length fillers are captured as gap patterns (e.g. `TRIP ~[0,3]→ RESTORE`), learned automatically from the covered array.
3. **Infer**: aligns the new sequence against the grammar via beam search; symbols not covered by any pattern contribute their Shannon bit cost to E.
4. **Score**: `e_norm = E / raw_cost`. 0 = all symbols covered by known patterns. 1 = nothing matched. Threshold default: any uncovered symbol (`e_norm > 0`) is anomalous — tune via `recalibrate`.
5. **Hierarchy**: patterns at level N reference patterns from level N-1, enabling detection of structural violations (wrong order, missing composition) that atom-level matching would miss.

## Documentation

- [Architecture](docs/architecture.md) — scoring objective, beam search, learning loop
- [Grammar model](docs/grammar-spec.md) — data model, GapConstraint, what was excluded and why
- [Scoring](docs/scoring.md) — E_norm, threshold semantics, per-level calibration
- [Performance](docs/performance.md) — current optimisations and improvement roadmap
- [Known limitations](docs/known-limitations.md) — what the current implementation cannot do

## Benchmark results

For now (2026), validated on LogHub HDFS: Precision=0.973, Recall=0.825, F1=0.893 — trained on 1k normal sequences, no labels, no embeddings, no feature vectors — see [spma-experiments](https://github.com/geronimo-iia/spma-experiments). There is still a lot to explore — broader datasets, deeper comparison, open questions.
