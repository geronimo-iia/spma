# Plan: Inverted index for hit detection (C)

## What it solves

In `beam_search` ([`spma/src/beam.rs`](../spma/src/beam.rs)), the current hit
detection builds a local `symbol_to_old` map on every call:

```rust
let mut symbol_to_old: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
for (oi, pat) in old.iter().enumerate() {
    for (pos, sym_ref) in pat.symbols.iter().enumerate() {
        let id = match sym_ref { ... };
        symbol_to_old.entry(id).or_default().push((oi, pos));
    }
}
```

This rebuilds the full index from scratch on every `beam_search` call — once
per sequence during training, once per sequence during infer. At 446k sequences
the cost is O(L × N × avg_pattern_len) per call where N = number of patterns.

The fix: build the index once when the grammar level is stable, reuse it across
all calls at that level.

---

## Baseline

No automated bench suite exists. Record wall-clock time manually before and
after implementation. Run from `spma-experiments/hdfs-validation/`.

**Training (50k corpus — shows train-path gain):**
```bash
time spma train \
  --corpus <(head -50000 data/preprocessed/train_normal.txt) \
  --output /tmp/hdfs_50000_bench.json \
  --beam 10
```

**Infer (446k sequences — shows infer-path gain):**
```bash
time spma infer \
  --model data/model/corpus/hdfs_50000.json \
  --corpus data/preprocessed/test_normal.txt > /dev/null
```

Pre-existing models at `data/model/corpus/hdfs_*.json`.
**Recorded baselines (feat/perf-g, release build, Apple Silicon):**
- Train 50k: `real 1m50s` (142s user, 129% CPU — sequential)
- Infer 446k: `real 2m4s` (1828s user, 1480% CPU — 16 cores)

Target: ≥2x wall-clock on both. Training is sequential so gain shows directly
in real time. Infer is parallel — user time drop matters more than real time.

---

## Where the index lives

The index belongs on `GrammarLevel` in
[`spma/src/model.rs`](../spma/src/model.rs), next to `patterns`:

```rust
pub struct GrammarLevel {
    pub patterns: Vec<Pattern>,
    pub symbol_index: SymbolIndex,   // new
}
```

`SymbolIndex` maps `symbol_id → Vec<(pattern_idx, position_in_pattern)>`:

```rust
pub struct SymbolIndex {
    // indexed by symbol_id; outer vec is sparse (symbol_id as index)
    occurrences: Vec<Vec<(u32, u16)>>,  // (pattern_idx, pos)
}
```

Using `pattern_idx` (position in `patterns` vec) rather than `pattern_id`
avoids a secondary lookup. `u16` for position is safe — no pattern has >65k
symbols.

**Do NOT serialize the index.** It is fully derivable from `patterns`.
Serializing it bloats `model.json` and risks stale-index bugs when `patterns`
mutate. Use `#[serde(skip)]` on the field; always rebuild in `Spma::load`.

---

## Steps

### 1. Add `SymbolIndex` to `model.rs`

- New struct `SymbolIndex` with `occurrences: Vec<Vec<(u32, u16)>>`
- `impl Default for SymbolIndex` — returns empty (needed for `#[serde(skip)]`
  deserialization default on existing saved models)
- Constructor `SymbolIndex::build(patterns: &[Pattern]) -> Self`
  — iterates patterns once, fills occurrences indexed by symbol id
- Method `fn get(&self, symbol_id: u32) -> &[(u32, u16)]`
  — returns empty slice if `symbol_id` out of bounds
- **No** `Serialize / Deserialize` — see note above

### 2. Update `GrammarLevel`

- Add `#[serde(skip)] symbol_index: SymbolIndex` field
  — `#[serde(skip)]` implies `Default` for deserialization; existing `model.json`
  files load without error, index starts empty and is rebuilt by `Spma::load`
- Update `GrammarLevel::new` to accept and store the index (or build it from
  patterns directly)
- Add `GrammarLevel::rebuild_index(&mut self)` — rebuilds from `self.patterns`,
  called after any mutation of `patterns`

### 3. Build the index at the right moments in `engine.rs`

Three places where `patterns` are finalized and the index must be built:

- **Cold-start training** (`train_inner`): after `self.grammar.levels.push(...)` for
  each level — call `rebuild_index` on the newly pushed level
- **Incremental training** (`train_inner` retrain path): after
  `self.grammar.levels[level].patterns.extend(...)` — call `rebuild_index`
- **`load`**: `#[serde(skip)]` gives an empty index; call `rebuild_index` on
  each level after deserialization in `Spma::load`

### 4. Update `beam_search` signature and body

Current signature:
```rust
pub fn beam_search(new: &[u32], old: &[&Pattern], beam_k: usize, costs: &[f64]) -> Vec<RawAlignment>
```

New signature:
```rust
pub fn beam_search(
    new: &[u32],
    old: &[&Pattern],
    index: &SymbolIndex,
    beam_k: usize,
    costs: &[f64],
) -> Vec<RawAlignment>
```

Body change — replace the local map build:
```rust
// remove this block entirely
let mut symbol_to_old: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
for (oi, pat) in old.iter().enumerate() { ... }
```

Replace the lookup:
```rust
// before
if let Some(matches) = symbol_to_old.get(&sym) {

// after
let matches = index.get(sym);
if !matches.is_empty() {
```

### 5. Update all `beam_search` call sites in `engine.rs`

Three call sites:
- Level-0 beam during training (`raw_results` parallel map)
- Level-0 beam during `infer`
- Higher-level beam during `infer` (the `for level in 1..` loop)

Each passes `&self.grammar.levels[level].symbol_index`.

### 6. Update tests

`beam_search` unit tests in `beam.rs` build `old: Vec<&Pattern>` directly
without a `GrammarLevel`. Add a test helper:

```rust
fn index_for(patterns: &[&Pattern]) -> SymbolIndex {
    let owned: Vec<Pattern> = patterns.iter().map(|p| (*p).clone()).collect();
    SymbolIndex::build(&owned)
}
```

One call per test, pass `&index_for(&old)` to `beam_search`.

---

## What does NOT change

- `PartialAlignment`, `MatchArena`, `MatchEvent` — untouched
- `can_extend` logic — untouched
- Beam scoring and truncation — untouched
- `GapConstraint` matching — untouched
- Public API (`Spma::infer`, `Spma::train`, `Spma::retrain`, `Spma::recalibrate`) — untouched

---

## Correctness check

The index maps `symbol_id → (pattern_idx, pos)`. The existing `can_extend`
check uses `old_idx` (= `pattern_idx`) and `old_pos` (= `pos`) — same
semantics. No logic change, only the source of the `(oi, q)` pairs changes.

Edge case: `SymbolRef::Pattern(id)` at higher grammar levels. The id there is a
pattern id, not a symbol id in the atom interner. The existing code already
handles this uniformly via the `match sym_ref` arm — the index must do the same,
treating both `Atom(id)` and `Pattern(id)` as raw `u32` keys.

---

## Serde compatibility

Existing saved models (`model.json`) have no `symbol_index` field.
`#[serde(skip)]` on the field means they deserialize to `SymbolIndex::default()`
(empty). `Spma::load` must call `rebuild_index` on each level after
deserialization before any `infer` call.

---

## Expected outcome

Per `performance.md`: expected 10–100x on large grammars.
The grammar saturates at ~147 patterns (9 levels) on HDFS — the gain is most
visible at infer time over 446k sequences, not at training time over 1k.
Record wall-clock `time spma infer` before and after against the 446k corpus.
