# Treat uncovered atoms as neutral (zero cost)

## Motivation

FP analysis on the HDFS 1k model shows 92% of false positives (359/389) are
caused by 5 atoms that never appear in any pattern: E6, E16, E18, E25, E28.
These atoms are rare in the training corpus — too rare for MDL to build patterns
around them — but they do appear in *normal* test sequences.

Current behavior: uncovered atom → atom_cost[id] added to e_cost → e_norm > 0
→ flagged as anomaly even if the rest of the sequence compresses perfectly.

This is wrong for the "rare-but-valid" case. If a sequence contains only
covered atoms and one rare-but-valid uncovered atom, it should not be flagged.

**Proposed fix:** atoms that appear in the training vocabulary but are not
covered by any pattern contribute 0 to e_cost instead of their atom_cost.

Only truly unknown atoms (not in interner at all, i.e. out-of-vocabulary)
retain their fallback_cost — those are genuinely novel symbols.

## Terminology

- **Uncovered atom**: in the interner (seen during training) but not referenced
  by any `SymbolRef::Atom` in any pattern across any level
- **Unknown atom**: not in the interner at all (out-of-vocabulary, new symbol)

## Files to touch

- `src/engine.rs` — `Spma::infer()` and `Spma` struct
- `src/model.rs` — optionally store uncovered set in model for fast lookup

## Implementation

### 1. Compute covered atom set

In `train()`, after all grammar levels are built, compute which atom IDs appear
in at least one pattern symbol:

```rust
let covered_atoms: std::collections::HashSet<u32> = self.grammar.levels
    .iter()
    .flat_map(|lvl| lvl.patterns.iter())
    .flat_map(|pat| pat.symbols.iter())
    .filter_map(|sym| if let SymbolRef::Atom(id) = sym { Some(*id) } else { None })
    .collect();
```

Store this set on `Spma`:

```rust
pub struct Spma {
    pub grammar: Grammar,
    beam_k: usize,
    pub atom_costs: Vec<f64>,
    max_induced_gap: usize,
    covered_atoms: std::collections::HashSet<u32>,  // add this
}
```

Initialize to empty in `new()`. Populate at end of `train()`.

**Serde:** add `#[serde(default)]` to `covered_atoms` so existing model JSON
files load without error. After loading an old model, call a helper to
recompute it:

```rust
pub fn recompute_covered_atoms(&mut self) {
    self.covered_atoms = self.grammar.levels
        .iter()
        .flat_map(|lvl| lvl.patterns.iter())
        .flat_map(|pat| pat.symbols.iter())
        .filter_map(|sym| if let SymbolRef::Atom(id) = sym { Some(*id) } else { None })
        .collect();
}
```

Call `recompute_covered_atoms()` inside `load()` after deserialization:

```rust
pub fn load<R: IoRead>(reader: R) -> io::Result<Self> {
    let mut spma: Self = serde_json::from_reader(reader)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    spma.recompute_covered_atoms();
    Ok(spma)
}
```

### 2. Build zero-cost costs vector in `infer()`

In `infer()`, replace the current `costs` construction:

```rust
// Current (wrong for uncovered atoms):
let fallback_cost = self.atom_costs.iter().cloned().fold(1.0f64, f64::max);
let mut costs: Vec<f64> = self.atom_costs.clone();
costs.resize(costs_len, fallback_cost);
```

With:

```rust
// New: uncovered atoms get 0 cost, only OOV atoms get fallback
let fallback_cost = self.atom_costs.iter().cloned().fold(1.0f64, f64::max);
let mut costs: Vec<f64> = self.atom_costs
    .iter()
    .enumerate()
    .map(|(id, &c)| if self.covered_atoms.contains(&(id as u32)) { c } else { 0.0 })
    .collect();
costs.resize(costs_len, fallback_cost);  // OOV slots keep fallback
```

This means:
- Covered atom: pays its atom_cost (existing behavior)
- Uncovered atom (in vocab, not in patterns): pays 0
- OOV atom (not in vocab): pays fallback_cost (existing behavior)

`raw_new_cost` and `e_cost` are computed from this `costs` vec — no other
changes needed. `e_norm = e_cost / raw_new_cost` handles the zero case via the
existing `raw_new_cost < 1e-12` guard.

### 3. HashSet serde

`HashSet<u32>` serializes fine with serde. Add `#[serde(default)]` only —
no custom serializer needed. Since `recompute_covered_atoms()` is called in
`load()`, the stored value in JSON is irrelevant (always recomputed).
Could skip serializing it entirely with `#[serde(skip)]` to keep model JSON clean.

Use `#[serde(skip)]` — simpler, model JSON unchanged:

```rust
#[serde(skip)]
covered_atoms: std::collections::HashSet<u32>,
```

Then `recompute_covered_atoms()` must always be called after load AND after
train. In `train()`, call it at the end. In `load()`, call it as shown above.

## Tests

Add to `tests/training.rs` or a new `tests/uncovered_atoms.rs`:

```rust
#[test]
fn uncovered_atom_does_not_flag_normal_sequence() {
    // Train on sequences that never contain token "rare"
    // Verify "rare" appears in interner (add it manually or via a single
    // training sequence without enough frequency to be covered)
    //
    // Simpler: train on ["A B C", "A B C", "A B C"] x many
    // Then infer "A B C X" where X is a new token — should NOT be anomaly
    // if X is treated as zero-cost (but X here is OOV, not uncovered)
    //
    // Better test: train on corpus where "R" appears once (below min_freq)
    // so it's in the interner but not covered. Infer "A B C R" — should
    // not flag if R is uncovered (zero cost).
    // Note: single occurrence may not get interned depending on train logic.
    // Check if interner.intern() is called unconditionally before MDL gating.
    // If yes: "R" appearing once IS in interner, but below min_freq for patterns.
    // Construct the test accordingly.
}

#[test]
fn oov_atom_still_flags_as_anomaly() {
    // Train on ["A B C"] x many
    // Infer "A B C Z" where Z was never seen during training
    // Z is OOV → fallback_cost → e_cost > 0 → is_anomaly = true
}
```

Adjust test construction to match actual `Spma` API (string sequences).

## Verification on HDFS

After implementing, run on 1k model:

```bash
cd hdfs-validation

# Regenerate infer results with new binary
spma infer --model data/model/corpus/hdfs_1000.json \
           --threshold 0.0 \
           --input data/splits/test_normal.txt \
           --json > /tmp/norm_new.jsonl || true

spma infer --model data/model/corpus/hdfs_1000.json \
           --threshold 0.0 \
           --input data/splits/test_anomaly.txt \
           --json > /tmp/anom_new.jsonl || true

python eval.py /tmp/norm_new.jsonl /tmp/anom_new.jsonl
```

Expected: FP drops from 389 → ~30 (the 30 non-uncovered-atom FP).
Precision should rise toward 0.998+. Recall unchanged (FN unaffected).

## Decision rule

Keep if:
- `cargo test` passes (including new tests)
- FP on 1k model drops significantly (target: < 50)
- TP unchanged (recall not regressed)
- F1 improves over current 0.893 baseline
