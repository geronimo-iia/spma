# Calibrated E-Score Design — Phase 1c

Specification for replacing the binary `is_anomaly: bool` threshold with a calibrated,
SP-theoretically grounded anomaly score. No compatibility constraint with current output.

Depends on: `docs/grammar-spec.md` (EPercentiles struct, GrammarLevel corpus_costs).

---

## SP theory grounding

E is the encoding cost in bits of the New symbols that the grammar could not cover. In
SP theory, E has intrinsic information-theoretic meaning: it is exactly the number of bits
required to transmit the uncovered part of the New sequence to a receiver who has the
grammar but not the sequence.

E = 0 means the grammar fully predicts the New sequence. No bits needed beyond the grammar
itself. Perfect alignment.

E > 0 means some symbols were unexpected — the grammar has no pattern that covers them,
so they must be transmitted raw.

### Why raw E is not a universal threshold

Raw E (bits) has two confounds that make it useless as a cross-sequence threshold:

1. **Sequence length**: a 20-symbol sequence has more uncovered symbols available than a
   2-symbol sequence. E = 4.0 bits on a length-2 sequence is catastrophic; on a length-20
   sequence it may be normal.

2. **Symbol cost distribution**: a grammar trained on a 5-symbol alphabet has cheap
   symbols (~2.3 bits each). A grammar trained on a 500-symbol alphabet has expensive
   symbols (~9 bits each). E = 4.0 bits means very different things across these grammars.

The fix for both confounds is **normalization**.

---

## Primary signal: normalized E (E_norm)

```
E_norm = E / raw_new_cost
```

Where `raw_new_cost` = sum of bit costs of all New symbols assuming no grammar coverage
(i.e. all symbols encoded raw).

**Properties**:
- Range: [0.0, 1.0]
- 0.0 = perfect coverage (grammar fully explains New)
- 1.0 = nothing matched (grammar covers nothing)
- Length-independent: dividing by raw_new_cost removes the sequence-length confound
- Cost-distribution-independent: dividing by the same cost function removes the
  alphabet-size confound
- Equivalent to `1.0 - (CD / raw_new_cost)` — the fraction of bits NOT saved by the grammar
- SP-theoretically meaningful: it is the compression failure rate of the grammar on this
  sequence

`E_norm` is the primary anomaly signal. The binary `is_anomaly` threshold becomes
`E_norm > threshold` where `threshold` is operator-configurable (default: see below).

### Default threshold

Default is `E_norm > 0.0` — any uncovered symbol = anomaly. Identical behavior to the
current `E > 0` check, now length- and cost-normalized. Operators can raise it via
`set_anomaly_threshold()` if they want to tolerate partial coverage. Do not silently
change the default to p90 — that flags 10% of training sequences as anomalous by design.

---

## Secondary signal: E percentile

Where does this `E_norm` value fall in the distribution of `E_norm` values observed
during training?

```
anomaly_percentile = fraction of training sequences with E_norm <= this E_norm
```

0.0 = lower E_norm than all training sequences (extremely well compressed).
0.97 = higher E_norm than 97% of training sequences.
1.0 = higher E_norm than any training sequence seen.

The percentile is computed by binary search into a sorted array of training E_norm values.
It complements E_norm: E_norm tells you the absolute compression failure rate,
percentile tells you where that falls relative to what the grammar has seen.

---

## Training: storing the distribution

During training, after the N-level loop converges, compute E_norm for every training
sequence using the final grammar. Store the sorted array.

```rust
pub struct EDistribution {
    /// E_norm values from training sequences, sorted ascending.
    sorted_e_norms: Vec<f64>,

    /// Anomaly gate: E_norm > threshold → is_anomaly.
    /// Default: 0.0 (any uncovered symbol = anomaly, same as current E>0 behavior).
    pub threshold: f64,

    /// Per-level distributions, one sorted vec per grammar level.
    pub level_sorted_e_norms: Vec<Vec<f64>>,
}
```

No pre-computed p50/p90/etc fields — derive with `partition_point` on demand. Operator
can call `e_distribution().percentile(0.9)` to get p90 if needed.

### Computing during training (engine.rs)

After the learning loop, iterate all training sequences through `infer_internal()` and
collect `E_norm` values:

```rust
let mut e_norms: Vec<f64> = training_sequences
    .iter()
    .map(|seq| {
        let result = self.infer_internal(seq);
        result.e_cost / result.raw_new_cost.max(1e-12)  // guard against zero-length
    })
    .collect();
e_norms.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
```

`infer_internal` is a lighter version of `infer` that skips alignment table construction.
No match log, no `AlignmentRow` allocation, no `CellContent` dispatch. Returns only E cost
and raw_new_cost.

```rust
// Private — engine.rs only.
fn infer_internal(&self, seq: &[u32]) -> (f64, f64) {
    // (e_cost, raw_new_cost)
    // Runs beam search without match_log.
    // raw_new_cost = sum of bit_cost for every symbol in seq (unconstrained cost).
    // e_cost = sum of bit_cost for uncovered positions after beam search.
    // Returns (0.0, 0.0) for empty sequences — caller must guard before dividing.
}
```

Zero-length sequences are excluded from the distribution (their E_norm is undefined —
division by zero). Guard: `if raw_new_cost < 1e-12 { return; }`.

---

## Inference: emitting the score

Authoritative `InferResult` definition is in `docs/alignment-struct-design.md`. Fields
relevant to calibration:

```rust
pub e_norm: f64,              // E / raw_new_cost, range [0.0, 1.0]
pub anomaly_percentile: f64,  // fraction of training sequences with e_norm <= this
pub level_e_norms: Vec<f64>,  // per-level e_norm contributions
pub is_anomaly: bool,         // e_norm > grammar.e_distribution.threshold
```

`is_anomaly` is now derived from `e_norm > grammar.e_distribution.threshold` rather than
`e_cost > 0.0`. Behavior is identical when threshold = 0.0 (the default), but operators
can raise it.

### Computing anomaly_percentile

```rust
fn percentile(dist: &EDistribution, e_norm: f64) -> f64 {
    if dist.sorted_e_norms.is_empty() {
        return 0.0;
    }
    let pos = dist.sorted_e_norms.partition_point(|&x| x <= e_norm);
    pos as f64 / dist.sorted_e_norms.len() as f64
}
```

`partition_point` is binary search — O(log n). For a training corpus of 10k sequences,
this is ~14 comparisons.

---

## Per-level calibration

`level_costs` from Fix B are raw bit costs, same confound as raw E. Normalize them too:

```rust
pub level_e_norms: Vec<f64>,  // level_costs[i] / raw_new_cost, one per grammar level
```

Each level has its own E_norm contribution. Useful for diagnosis: a sequence might have
`e_norm = 0.0` at level 0 (atom patterns cover everything) but `level_e_norms[0] = 0.4`
at level 1 (ordering anomaly — the right atoms appear but in the wrong composition).

This is the key SP-theoretic insight: a sequence can be syntactically correct at one level
of abstraction and anomalous at another. The per-level E_norm exposes exactly which level
of the grammar the sequence violated.

Per-level calibration uses the same `EDistribution` approach: store sorted per-level
`E_norm` values from training, compute percentile at inference.

**Per-level denominator**: `level_e_norms[i] = level_costs[i] / raw_level_i_cost` where
`raw_level_i_cost` is the raw encoding cost of the **pattern-ID sequence at level i** —
NOT the original atom sequence. At level 1, the input is a sequence of level-0 pattern
IDs; each ID has its own bit cost based on its frequency in the level-0 alignment. Using
the atom-level `raw_new_cost` as denominator for level-1 E_norm would be incorrect —
the reference cost is the cost of the level-i input, not the original sequence.

---

## Operator API

```rust
impl Spma {
    /// Set the E_norm threshold above which is_anomaly = true.
    /// Default: 0.0 (any uncovered symbol = anomaly — same as original E>0 behavior).
    pub fn set_anomaly_threshold(&mut self, threshold: f64);

    /// Returns the E_norm distribution from training (for custom thresholding).
    pub fn e_distribution(&self) -> &EDistribution;
}
```

---

## What is NOT in scope for 1c

- Per-sequence E_norm calibration across different grammars (cross-grammar comparison
  requires a common reference corpus — out of scope)
- Online distribution update as new sequences arrive (would require incremental sort —
  out of scope, full retrain for now)
- Probabilistic score in the frequentist/Bayesian sense — E_norm is information-theoretic,
  not a probability. Do not call it a probability. The percentile is a rank statistic.

---

## Serialization

`EDistribution` serializes as part of `GrammarFile`:

```rust
// In GrammarFile (MessagePack schema):
struct EDistributionRecord {
    sorted_e_norms: Vec<f64>,
    threshold: f64,
    // per-level distributions, one entry per grammar level
    level_sorted_e_norms: Vec<Vec<f64>>,
}
```

`sorted_e_norms` is O(n_training_sequences) floats — at 10k sequences this is 80KB
uncompressed, acceptable for a grammar file. MessagePack float64 arrays compress well
(~40% with zstd if that's added later).

If training corpus is large (>100k sequences), subsample to 10k for the distribution
(random sample, preserve rank statistics within 1-2% error). Store the subsample size
alongside so callers know the resolution.
