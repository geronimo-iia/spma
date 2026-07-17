# Task: Add `--threshold` override flag to `spma infer`

## Context

`spma train` stores `threshold` in `grammar.e_distribution.threshold` inside the
model JSON. At inference time (`src/engine.rs` line 491):

```rust
let is_anomaly = e_norm > self.grammar.e_distribution.threshold;
```

The grammar, `e_norm`, and `EDistribution` percentiles are all independent of
threshold — only this comparison uses it. So overriding threshold at infer time
produces identical results to training with that threshold.

This enables threshold sweep as **1 train + N infer** instead of **N train + N infer**.

## Change

**File: `src/bin/spma.rs` only. No other files.**

Add to `Command::Infer`:

```rust
/// Override anomaly threshold from model (e_norm cutoff).
/// If omitted, uses value stored in the model.
#[arg(short, long)]
threshold: Option<f64>,
```

In the `Command::Infer` match arm, after loading the model and before the infer
loop, add:

```rust
if let Some(t) = threshold {
    spma.set_anomaly_threshold(t);
}
```

`set_anomaly_threshold` already exists in `src/engine.rs`:
```rust
pub fn set_anomaly_threshold(&mut self, threshold: f64) {
    self.grammar.e_distribution.threshold = threshold;
}
```

That's the entire change.

## Verify

```bash
cargo build --release

# Train once
./target/release/spma train \
  --corpus /tmp/hdfs_50k.txt \
  --output /tmp/hdfs_base.json \
  --beam 10

# Infer with threshold override
./target/release/spma infer \
  --model /tmp/hdfs_base.json \
  --input /tmp/hdfs_1k.txt \
  --threshold 0.5 \
  --json > /tmp/infer_override.jsonl

# Train with same threshold, infer without override
./target/release/spma train \
  --corpus /tmp/hdfs_50k.txt \
  --output /tmp/hdfs_t05.json \
  --threshold 0.5 \
  --beam 10
./target/release/spma infer \
  --model /tmp/hdfs_t05.json \
  --input /tmp/hdfs_1k.txt \
  --json > /tmp/infer_trained.jsonl

# Must be identical
diff /tmp/infer_override.jsonl /tmp/infer_trained.jsonl
```

`diff` must be empty.

Also verify no-flag behavior unchanged (model threshold used when `--threshold` omitted):

```bash
./target/release/spma infer \
  --model /tmp/hdfs_base.json \
  --input /tmp/hdfs_1k.txt \
  --json > /tmp/infer_default.jsonl

# Compare to same binary without flag — must match
diff /tmp/infer_default.jsonl /tmp/infer_default.jsonl  # trivially, but run the command
```

`cargo test` must pass. `cargo clippy` must pass.

## Commit

```
feat(cli): add --threshold override flag to spma infer
```

## Impact on threshold.sh

After this change, update
`/Users/geronimo/dev/projects/libraries/spma-experiments/hdfs-validation/threshold.sh`
to train once and sweep threshold at infer time:

```bash
SPMA=/Users/geronimo/dev/projects/libraries/spma/target/release/spma
DIR="$(cd "$(dirname "$0")" && pwd)"
CORPUS=/tmp/hdfs_50k.txt

echo "=== training base model ==="
$SPMA train --corpus "$CORPUS" --output "$DIR/data/hdfs_base.json" --beam 10

for T in 0.0 0.1 0.2 0.3 0.5 0.7 1.0; do
  $SPMA infer --model "$DIR/data/hdfs_base.json" \
              --threshold "$T" \
              --input "$DIR/data/test_normal.txt" \
              --json > "$DIR/data/results_normal_t${T}.jsonl"

  $SPMA infer --model "$DIR/data/hdfs_base.json" \
              --threshold "$T" \
              --input "$DIR/data/test_anomaly.txt" \
              --json > "$DIR/data/results_anomaly_t${T}.jsonl" || true

  echo "=== T=$T ===" && python "$DIR/eval.py" \
    "$DIR/data/results_normal_t${T}.jsonl" \
    "$DIR/data/results_anomaly_t${T}.jsonl"
done
```
