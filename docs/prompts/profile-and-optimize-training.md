# Task: Profile and optimize SPMA training speed

## Context

`spma train` on 1000 HDFS sequences (avg ~15 tokens, vocab E1–E30) takes 9.4s.
Extrapolates to ~70 min for 446k sequences. Need to identify bottlenecks and fix.

Codebase: `/Users/geronimo/dev/projects/libraries/spma/`

Key files:
- `src/engine.rs` — `train()`, main loop at line 216 (`for level in 0..max_levels`)
- `src/beam.rs` — `beam_search()`, called per sequence per level
- `src/alignment.rs` — alignment construction

## Phase 1: Instrument to find bottleneck

Add wall-clock timers (using `std::time::Instant`) inside `train()` to measure
each major section. Print to stderr when a `SPMA_PROFILE=1` env var is set.
Do NOT add a CLI flag for this — env var only, gated with `std::env::var("SPMA_PROFILE").is_ok()`.

Sections to time independently:

```
[profile] step3_ngrams:       Xms   (extract_frequent_ngrams + MDL gate cold-start)
[profile] level0_mdl:         Xms   (MDL gating of level0 patterns)
[profile] levelN_beam_total:  Xms   (all beam_search calls across all levels, cumulative)
[profile] levelN_gap_extract: Xms   (extract_learned_patterns calls, cumulative)
[profile] levelN_mdl:         Xms   (MDL gating of next-level patterns, cumulative)
[profile] edist_rebuild:      Xms   (EDistribution pass at end of train, lines 377–500)
```

For `levelN_*` timers: accumulate across all levels, print once at end.
Also print per-level breakdown:

```
[profile] level 0 beam: Xms  gap: Xms  mdl: Xms  patterns_in: N  patterns_out: M
[profile] level 1 beam: Xms  gap: Xms  mdl: Xms  patterns_in: N  patterns_out: M
...
```

Run with:
```bash
SPMA_PROFILE=1 cargo run --release --bin spma -- train \
  --corpus /tmp/hdfs_1k.txt \
  --output /tmp/hdfs_1k.json \
  --beam 10
```

**Report the profile output in full before doing any optimization.**

## Phase 2: Identify hot path

Based on profile output, determine which section dominates. Expected suspects:

### Suspect A — `compute_total_e_dp` inside MDL gate (O(P × N) per candidate)

In `engine.rs` around line 352 and 358:
```rust
let current_e = compute_total_e_dp(&pid_seqs, &next_id_vecs, &pid_costs);
// ...
let new_e = compute_total_e_dp(&pid_seqs, &candidate, &pid_costs);
```

Called once per candidate pattern per level. If there are P candidates and N
sequences, this is O(P × N × seq_len) per level. Find `compute_total_e_dp` in
`src/engine.rs` or `src/alignment.rs`, understand its complexity, report it.

### Suspect B — `beam_search` per sequence per level

Called `N_sequences × N_levels` times. With 446k sequences and 9 levels = 4M calls.
Inspect `src/beam.rs` — what is the complexity per call? Does it allocate heavily?

### Suspect C — `extract_learned_patterns` per sequence per level

Called per sequence in the inner loop. Check allocation pattern.

### Suspect D — `edist_rebuild` — full re-run of beam on all sequences after training

Lines 377–500: runs beam again on all 446k sequences just to compute e_norm
distribution. This is a full second pass = same cost as one training level.

## Phase 3: Implement fixes (ordered by expected impact)

Implement only fixes confirmed as significant by profile. Suggested approaches:

### Fix 1 — Skip `edist_rebuild` full pass: reuse training match logs

The e_norms pass (lines 377–500) re-runs beam on all sequences after training.
But match logs from the training loop already contain alignment data.
Cache the final-level match log per sequence during the training loop and reuse it.

Specifically: during Step 4 loop, at the last level, save `e_cost` per sequence.
After the loop, use those saved values instead of re-running beam.

This eliminates one full pass over all sequences.

### Fix 2 — Incremental MDL: cache `current_e`, update instead of recompute

Inside the MDL gate loop (around line 352):
```rust
let current_e = compute_total_e_dp(&pid_seqs, &next_id_vecs, &pid_costs);
```

`current_e` doesn't change between iterations when `new_t >= current_t` (pattern
rejected). Cache it and only recompute after an acceptance:

```rust
let mut cached_e = compute_total_e_dp(&pid_seqs, &next_id_vecs, &pid_costs);
for pat in next_level_pats {
    // use cached_e as current_e
    // only update: cached_e = new_e  when accepted
}
```

### Fix 3 — Parallelize beam_search across sequences (rayon)

The per-sequence beam_search calls are independent. Add `rayon` to `Cargo.toml`
and use `par_iter()` on the sequence batch:

```toml
[dependencies]
rayon = "1"
```

```rust
use rayon::prelude::*;

let raw_results: Vec<Option<RawAlignment>> = current_atom_seqs
    .par_iter()
    .map(|seq| {
        beam_search(seq, &level_patterns, self.beam_k, &current_costs)
            .into_iter()
            .next()
    })
    .collect();
```

Note: frequency update loop after this must stay sequential (mutable borrow on
`self.grammar.levels[level].patterns`). The parallelism only covers the
read-only beam pass.

### Fix 4 — Reduce allocations in `beam_search`

Profile Suspect B — if `beam_search` allocates a Vec per call, pre-allocate a
reusable buffer. Only implement if profiling shows it significant.

## Phase 4: Measure speedup

After each fix, re-run:
```bash
time cargo run --release --bin spma -- train \
  --corpus /tmp/hdfs_1k.txt \
  --output /tmp/hdfs_1k.json \
  --beam 10
```

Report before/after times. Target: 1000 sequences in < 2s (5× speedup).

Then validate correctness — inference results must not change:
```bash
# Before fix: save reference output
cargo run --release --bin spma -- infer \
  --model /tmp/hdfs_1k_before.json \
  --input /tmp/hdfs_sample10.txt \
  --json > /tmp/before.jsonl

# After fix:
cargo run --release --bin spma -- infer \
  --model /tmp/hdfs_1k_after.json \
  --input /tmp/hdfs_sample10.txt \
  --json > /tmp/after.jsonl

diff /tmp/before.jsonl /tmp/after.jsonl
```

`is_anomaly` values must be identical. `e_cost`/`e_norm` may differ by < 1e-6
(float rounding). Any larger divergence = correctness regression, revert that fix.

## Constraints

- No changes to public API (`Spma::train`, `Spma::infer`, `InferResult` fields)
- No changes to test files
- `cargo test` must pass before and after
- `cargo clippy` must pass (no new warnings)

## Commit

One commit per fix:
```
perf(engine): cache current_e across MDL gate iterations
perf(engine): parallelize beam_search with rayon
perf(engine): reuse training match logs for e-distribution pass
```
