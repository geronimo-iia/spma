# Scoring

## E_norm — normalized encoding cost

```
E_norm = E / raw_new_cost
```

`raw_new_cost` = sum of bit costs of all New symbols as if none were covered.

**Range**: [0.0, 1.0].  
0.0 = grammar fully covers the sequence.  
1.0 = nothing matched.

Raw E (bits) is not usable as a threshold across sequences: a 20-symbol sequence can have E=4.0 normally; on a 2-symbol sequence it is catastrophic. Dividing by `raw_new_cost` removes both the sequence-length and alphabet-size confounds. Result is the compression failure rate of the grammar on this sequence.

## Threshold

`is_anomaly = E_norm > threshold`. Default: 0.0 (any uncovered symbol = anomaly).

Operators can raise it via `Spma::set_anomaly_threshold()` to tolerate partial coverage. Do not default to p90 — that flags 10% of training sequences as anomalous by design.

## anomaly_percentile

Fraction of training sequences with `E_norm <= this E_norm`. Binary search into sorted training distribution. 0.0 = lower than all training sequences. 1.0 = higher than any.

Complements E_norm: E_norm is the absolute compression failure rate; percentile is its rank relative to what the grammar has seen.

## EDistribution

Stored on the grammar after training:

```rust
pub struct EDistribution {
    sorted_e_norms: Vec<f64>,       // from training corpus, sorted ascending
    pub threshold: f64,             // anomaly gate
    pub level_sorted_e_norms: Vec<Vec<f64>>,  // per grammar level
}
```

No pre-computed percentile fields. Compute with `partition_point` on demand.

## Per-level E_norm

`level_e_norms[i] = level_costs[i] / raw_level_i_cost`

The denominator is the raw encoding cost of the **level-i input** (pattern-ID sequence at that level), not the original atom sequence. Using atom-level cost as denominator for level-1 would be incorrect.

Key diagnostic: a sequence can have `e_norm = 0.0` at the atom level (all atoms covered) but `level_e_norms[1] > 0` at level 1 (ordering anomaly — correct atoms, wrong composition). The per-level E_norm exposes which grammar level was violated.

## What E_norm is NOT

E_norm is information-theoretic, not probabilistic. Do not call it a probability. The percentile is a rank statistic, not a confidence. Cross-grammar comparison requires a common reference corpus — out of scope.
