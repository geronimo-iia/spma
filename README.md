# spma

Unsupervised anomaly detection for discrete event sequences via MDL-based grammar induction (SP Multiple Alignment).

Learns a hierarchical grammar from normal sequences, then scores new sequences by their encoding cost E. Sequences that compress poorly — high `e_norm` — are anomalous. Validated on LogHub HDFS achieving **F1 = 0.893** unsupervised (see [spma-experiments](https://github.com/geronimo-iia/spma-experiments)).

## Install

```toml
[dependencies]
spma = "0.1"
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
| `cd` | `f64` | Compression difference; positive = grammar compresses the sequence |
| `level_costs` | `Vec<f64>` | Encoding cost per grammar level |
| `level_e_norms` | `Vec<f64>` | Normalised cost per grammar level |
| `alignment` | `Alignment` | Match table — which grammar patterns covered which symbols |

## Persist a model

```rust
use std::io::BufWriter;
use std::fs::File;

// Save
model.save(BufWriter::new(File::create("model.json")?));

// Load
let loaded = Spma::load(std::io::BufReader::new(File::open("model.json")?))?;
```

## Recalibrate thresholds

```rust
// After retraining or collecting more normal data:
model.recalibrate(&new_corpus);
```

## CLI quickstart

```sh
# Train
spma train --corpus normal.txt --output model.json

# Infer (exits 1 if any sequence is anomalous)
spma infer --model model.json --input sequences.txt
spma infer --model model.json --json < sequences.txt

# Refit e_distribution without re-training
spma recalibrate --model model.json --corpus new_normal.txt

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

```sh
spma infer --model model.json --level-threshold 0:0.2 --level-threshold 1:0.5
```

## How it works

1. **Train**: extracts frequent n-grams, then runs MDL-gated beam search to build a hierarchical grammar of patterns.
2. **Infer**: aligns the new sequence against the grammar; symbols not covered by any pattern contribute their Shannon bit cost to E.
3. **Score**: `e_norm = E / raw_cost`. 0 = all symbols covered by known patterns. 1 = nothing matched.

Scoring objective: **T = G + E** (MDL). G is charged once per pattern per insertion; E is the residual encoding cost.

## Benchmark results

See [spma-experiments](https://github.com/geronimo-iia/spma-experiments) for full results including HDFS LogHub evaluation (F1 = 0.893, unsupervised, no labeled data used during training).
