# Add `spma recalibrate` subcommand

## Goal

Add a `recalibrate` CLI subcommand that:
1. Loads an existing model JSON
2. Replays the training corpus through the **frozen grammar** (no re-induction)
3. Recomputes `EDistribution` (e_norms, level_e_norms, threshold) only
4. Saves the updated model back to disk

This enables a workflow where patterns are manually pruned from the model JSON
(e.g. LLM-guided removal of domain-invalid patterns), then the e_distribution
is recalibrated without touching the grammar structure.

## Why not just retrain?

Retraining re-induces the grammar from scratch — pruned patterns may reappear if
the corpus supports them. Recalibrate freezes the grammar and only replays the
corpus to refit the anomaly score distributions.

## Files to touch

- `src/bin/spma.rs` — add `Recalibrate` variant to `Command` enum, implement handler
- `src/engine.rs` — add `pub fn recalibrate(&mut self, corpus: &[Vec<&str>])` method

## Implementation

### 1. `engine.rs` — add `recalibrate`

Add a public method on `Spma` that replays corpus through existing grammar
and recomputes `e_distribution`. The grammar (`self.grammar.levels`,
`self.grammar.interner`, `self.atom_costs`) must NOT be modified.

The method body is identical to the final phase of `train()` that computes
`EDistribution::fit(...)`, but skips all induction steps.

```rust
pub fn recalibrate(&mut self, corpus: &[Vec<&str>]) {
    let mut e_norms: Vec<f64> = Vec::with_capacity(corpus.len());
    let mut level_e_norms_vecs: Vec<Vec<f64>> = Vec::new();

    for seq in corpus {
        let result = self.infer(seq);
        e_norms.push(result.e_norm);
        // result.level_e_norms is Vec<f64>, one entry per grammar level
        for (i, &lvl_e) in result.level_e_norms.iter().enumerate() {
            if level_e_norms_vecs.len() <= i {
                level_e_norms_vecs.push(Vec::new());
            }
            level_e_norms_vecs[i].push(lvl_e);
        }
    }

    self.grammar.e_distribution =
        crate::model::EDistribution::fit(e_norms, 0.0, level_e_norms_vecs);
}
```

Key details:
- `InferResult.level_e_norms` is `Vec<f64>` — one value per grammar level
- `EDistribution::fit` signature: `fit(e_norms: Vec<f64>, threshold: f64, level_e_norms: Vec<Vec<f64>>)`
- Grammar levels/interner/atom_costs must NOT be touched
- After fit, threshold is 0.0 — caller overrides via `--threshold` if needed

### 2. `spma.rs` — add `Recalibrate` variant

```rust
/// Reload a model, replay corpus to refit e_distribution, save updated model
Recalibrate {
    /// Path to saved model (modified in place or written to --output)
    #[arg(short, long)]
    model: String,

    /// Training corpus to replay (one sequence per line)
    #[arg(short, long)]
    corpus: String,

    /// Output path; if omitted, overwrites --model
    #[arg(short, long)]
    output: Option<String>,

    /// Override anomaly threshold after recalibration
    #[arg(short, long)]
    threshold: Option<f64>,
},
```

Handler:
1. Load model from `--model`
2. Read corpus with existing `read_corpus()`
3. Call `spma.recalibrate(&corpus_refs)`
4. Apply `--threshold` if provided
5. Save to `--output` (or `--model` if output is None)
6. Print to stderr: `recalibrated: N sequences, threshold={:.4}`

### 3. Tests

Add to `tests/` or inline in `engine.rs` under `#[cfg(test)]`:

```rust
#[test]
fn recalibrate_does_not_change_grammar_structure() {
    // Train a small model on normal corpus
    // Capture: levels.len(), patterns per level, interner names
    // Call recalibrate() with same corpus
    // Assert: grammar structure unchanged, e_distribution updated (not all zeros)
}

#[test]
fn recalibrate_after_pattern_removal() {
    // Train model, manually remove one pattern from levels[0].patterns
    // Call recalibrate() — must not panic or re-add the removed pattern
    // Assert: grammar still has N-1 patterns at level 0
}
```

## Verification

```bash
cargo test
cargo build --release

# Basic round-trip
./target/release/spma recalibrate \
    --model hdfs-validation/data/model/hdfs_base.json \
    --corpus hdfs-validation/data/splits/train_normal.txt \
    --output /tmp/hdfs_recalibrated.json

# Confirm grammar structure unchanged, distribution changed
./target/release/spma grammar --model hdfs-validation/data/model/hdfs_base.json
./target/release/spma grammar --model /tmp/hdfs_recalibrated.json
```

The comparison output should show identical level/pattern counts, potentially
different e_norm distributions if corpus subset differs.

## Decision rule

Keep `recalibrate` only if:
- It does not regress any existing test
- Round-trip recalibrate on same corpus produces threshold within 1% of original
