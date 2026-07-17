# Task: Per-level anomaly gate to reduce FN

## Motivation — SP theory

SP (Simplicity and Power) theory explains cognition and pattern recognition via
MDL: a sequence is anomalous when it cannot be compressed by learned grammar.
SPMA implements this as hierarchical multiple alignment — atoms at level 0,
pattern-IDs at level 1, pattern-of-pattern-IDs at level 2, etc.

The current `is_anomaly` gate uses a single `e_norm` aggregated at the atom
level:

```rust
// engine.rs line 491
let is_anomaly = e_norm > self.grammar.e_distribution.threshold;
```

This misses a class of anomalies that SP theory explicitly predicts: sequences
whose **atom-level alignment looks normal** (low `e_norm` at level 0) but whose
**pattern sequence is structurally abnormal** (high `e_norm` at level 1+).

### HDFS example

Normal HDFS block write: `[E22, E5, E11, E9, E26]` — allocate, receive, size,
ack, store. A corrupted sequence might use all the same atoms but in wrong order:
`[E5, E11, E22, E9, E26]` — atoms all covered (E_norm_level0 ≈ 0), but the
pattern-ID sequence violates the learned order grammar (E_norm_level1 > 0).
Current gate: not anomaly. Per-level gate: anomaly.

### SP theory prescription

Compute `e_norm` independently at each grammar level. Flag as anomaly if **any**
level exceeds its threshold. Each level captures a different granularity of
structural violation — atom coverage, pattern sequence order, higher-order
compositional structure.

## What already exists

All infrastructure is in place — this is wiring, not new computation:

- `InferResult.level_e_norms: Vec<f64>` — per-level e_norm, computed at every
  infer call (`engine.rs` lines 496–611)
- `EDistribution.level_sorted_e_norms: Vec<Vec<f64>>` — per-level training
  distributions (populated at train time, `model.rs` line 171)
- `EDistribution.level_percentile(level, e_norm)` — percentile lookup per level
  (`model.rs` line 199)

Missing: per-level threshold storage and the multi-level gate in `is_anomaly`.

## Change spec

### 1. `src/model.rs` — add `level_thresholds` to `EDistribution`

```rust
pub struct EDistribution {
    sorted_e_norms: Vec<f64>,
    pub threshold: f64,
    pub level_sorted_e_norms: Vec<Vec<f64>>,
    /// Per-level anomaly thresholds. Same length as level_sorted_e_norms.
    /// If empty or shorter than level_e_norms, falls back to global threshold.
    pub level_thresholds: Vec<f64>,
}
```

Update `EDistribution::default()` — add `level_thresholds: Vec::new()`.

Update `EDistribution::fit()` — add `level_thresholds: Vec::new()` (populated
separately via `set_level_threshold`).

Update `Serialize`/`Deserialize` — field is `#[serde]` auto-derived, no change
needed. Existing models without this field deserialize with `Vec::new()` via
`#[serde(default)]`:

```rust
#[serde(default)]
pub level_thresholds: Vec<f64>,
```

### 2. `src/engine.rs` — `set_level_threshold` and multi-level gate

Add method to `Spma`:

```rust
/// Set anomaly threshold for a specific grammar level.
/// Level 0 = atom level, level 1 = pattern-ID level, etc.
pub fn set_level_threshold(&mut self, level: usize, threshold: f64) {
    let dist = &mut self.grammar.e_distribution;
    if dist.level_thresholds.len() <= level {
        dist.level_thresholds.resize(level + 1, dist.threshold);
    }
    dist.level_thresholds[level] = threshold;
}
```

Replace the single-level gate in `infer()` (line ~491):

```rust
// OLD:
let is_anomaly = e_norm > self.grammar.e_distribution.threshold;

// NEW:
let dist = &self.grammar.e_distribution;
let is_anomaly_level0 = e_norm > dist.threshold;
let is_anomaly_levels = level_e_norms.iter().enumerate().any(|(lvl, &lvl_e)| {
    let t = dist.level_thresholds.get(lvl).copied().unwrap_or(dist.threshold);
    lvl_e > t
});
let is_anomaly = is_anomaly_level0 || is_anomaly_levels;
```

Note: `level_e_norms` is computed before this gate (lines 496–611 build it).
Restructure if needed so `level_e_norms` is available at gate time.

### 3. `src/bin/spma.rs` — `--level-threshold` flag on infer

```rust
/// Set per-level threshold as "level:value" pairs, e.g. --level-threshold 1:0.3
/// Can be specified multiple times. Falls back to global --threshold if not set.
#[arg(long, value_parser = parse_level_threshold)]
level_threshold: Vec<(usize, f64)>,
```

Parser:

```rust
fn parse_level_threshold(s: &str) -> Result<(usize, f64), String> {
    let (l, v) = s.split_once(':')
        .ok_or_else(|| format!("expected level:value, got {s:?}"))?;
    let level = l.parse::<usize>().map_err(|e| e.to_string())?;
    let value = v.parse::<f64>().map_err(|e| e.to_string())?;
    Ok((level, value))
}
```

In infer match arm, after loading model and applying `--threshold`:

```rust
for (level, t) in &level_threshold {
    spma.set_level_threshold(*level, *t);
}
```

## Backward compatibility

- No `--level-threshold` → behaviour identical to today (only global threshold used)
- Old models without `level_thresholds` deserialize cleanly via `#[serde(default)]`
- `level_thresholds` empty → `get(lvl)` returns `None` → falls back to `dist.threshold`

## Tests

Add to `tests/calibrated_score.rs`:

**`per_level_threshold_catches_missed_anomaly`**

Construct a scenario where level-0 e_norm is 0 (all atoms covered) but level-1
e_norm is high (pattern sequence not in grammar). Set `level_threshold[1]` low.
Assert `is_anomaly = true`. Assert same sequence with only global threshold = false.

```rust
// Hint: train on [A B C] [A B C], then infer [C B A] — atoms same, order inverted.
// Level-0: atoms A,B,C all covered → e_norm_0 ≈ 0
// Level-1: pattern sequence violates learned order → e_norm_1 > 0
```

**`per_level_threshold_fallback_to_global`**

`level_thresholds` empty → `is_anomaly` uses global threshold only. Assert
`set_level_threshold` extends vec correctly.

## Constraints

- `cargo test` passes (78 existing tests + new ones)
- `cargo clippy` clean
- `spma infer --help` shows `--level-threshold`
- No changes to `InferResult` public fields (they already have `level_e_norms`)

## Experiment usage (HDFS)

After implementation, sweep level-1 threshold independently:

```bash
for T1 in 0.0 0.1 0.2 0.3; do
  spma infer --model data/hdfs_base.json \
             --threshold 0.2 \
             --level-threshold 1:$T1 \
             --input data/test_normal.txt \
             --json > /tmp/norm_l1_${T1}.jsonl || true
  spma infer --model data/hdfs_base.json \
             --threshold 0.2 \
             --level-threshold 1:$T1 \
             --input data/test_anomaly.txt \
             --json > /tmp/anom_l1_${T1}.jsonl || true
  echo "T=0.2 L1=$T1:" && python eval.py /tmp/norm_l1_${T1}.jsonl /tmp/anom_l1_${T1}.jsonl
done
```

Expected: recall improves (FN caught at level 1) with minimal precision cost.

## Commit

```
feat(engine): per-level anomaly gate — catch structurally anomalous sequences missed at atom level
```
