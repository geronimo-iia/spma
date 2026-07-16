# SPMA Roadmap

Clean-slate implementation. Old code moved to `src_old/`. No compatibility constraint.
Design docs are authoritative — change docs first, code follows.

---

## Phase 0 — Scaffold (start here)

create the new module skeleton.

```
src/
  lib.rs        re-exports only
  intern.rs     copy from src_old (unchanged, no bugs)
  model.rs      Phase 1a
  beam.rs       Phase 1b
  alignment.rs  Phase 1b
  engine.rs     Phase 1d + 1e
```

`Cargo.toml`: drop `bincode`. Add `serde`, `serde_json`. Nothing else yet.

Done when: `cargo build` passes on an empty skeleton.

---

## Phase 1a — Data model (`model.rs`)

**Spec**: `docs/grammar-spec.md`

Structs to implement:

```rust
pub enum SymbolRef { Atom(u32), Pattern(u32) }
pub struct GapConstraint { pub min: usize, pub max: usize }
pub struct Pattern { pub id: u32, pub symbols: Vec<SymbolRef>, pub gaps: Vec<GapConstraint>, pub frequency: u32, pub level: u8 }
pub struct GrammarLevel { pub patterns: Vec<Pattern>, pub corpus_e_norms: Vec<f64> }
pub struct Grammar { pub interner: Interner, pub levels: Vec<GrammarLevel>, pub e_distribution: EDistribution }
pub struct EDistribution { pub sorted_e_norms: Vec<f64>, pub threshold: f64, pub level_sorted_e_norms: Vec<Vec<f64>> }
```

Rules:
- `Pattern::gaps` is empty for contiguous patterns. `gaps.len() == symbols.len() - 1` when non-empty. No other lengths valid.
- `EDistribution::threshold` default = `0.0`. Do not default to p90.
- No `Symbol` wrapper, no `SymbolType`, no `SymbolStatus`, no `SymbolRole`, no `AlignmentType`. Gone.
- `serde` derives on all structs. No custom serialization logic yet.

Done when: structs compile, `Pattern::new_contiguous` and `Pattern::new_with_gaps` constructors exist and enforce the `gaps.len()` invariant with `debug_assert`.

---

## Phase 1b — Beam + RawAlignment (`beam.rs`)

**Spec**: `docs/alignment-struct-design.md`

### MatchArena

```rust
struct MatchNode { event: MatchEvent, parent: Option<u32> }
struct MatchArena { nodes: Vec<MatchNode> }
impl MatchArena {
    fn push(&mut self, event: MatchEvent, parent: Option<u32>) -> u32
    fn collect(&self, tail: Option<u32>) -> Vec<MatchEvent>
}
```

Arena is owned by the `beam_search` stack frame. `PartialAlignment` holds `log_tail: Option<u32>`, not `Vec<MatchEvent>`. Forking copies one `u32`. No clone of the log on any beam step.

### PartialAlignment

```rust
struct PartialAlignment {
    old_cursors: HashMap<usize, usize>,   // last matched old_pos per old_idx
    new_cursors: HashMap<usize, usize>,   // last matched new_pos per old_idx
    max_covered_new: usize,
    covered_new: Vec<bool>,
    cd: f64,
    log_tail: Option<u32>,
}
```

`extend_match`: calls `arena.push(event, self.log_tail)`, stores returned index in `log_tail`.
`extend_skip`: copies `log_tail` unchanged — zero allocation.

### can_extend

Contiguous only for now (gap matching is Phase 2a):

```rust
fn can_extend(&self, old_idx: usize, old_pos: usize, new_pos: usize) -> bool {
    match self.old_cursors.get(&old_idx) {
        None => old_pos == 0 && new_pos >= self.max_covered_new,
        Some(&prev_old) => {
            if old_pos > prev_old {
                // Advancing within pattern — New must be contiguous.
                self.new_cursors
                    .get(&old_idx)
                    .map_or(false, |&prev_new| new_pos == prev_new + 1)
            } else {
                // old_pos == 0: fresh start of same pattern at new position.
                old_pos == 0 && new_pos >= self.max_covered_new
            }
        }
    }
}
```

### Output

```rust
pub struct MatchEvent { pub old_idx: usize, pub old_pos: usize, pub new_pos: usize, pub cost: f64 }
pub struct RawAlignment { pub match_log: Vec<MatchEvent>, pub covered: Vec<bool>, pub e_cost: f64, pub cd: f64 }

pub fn beam_search(new: &[u32], old: &[&Pattern], beam_k: usize, costs: &[f64]) -> Vec<RawAlignment>
```

`finalize()` calls `arena.collect(winning.log_tail)` once for the winning alignment only.

Done when: existing beam tests ported and passing. `MatchArena` unit test: fork two branches from same parent, verify both collect correct independent logs.

---

## Phase 1c — Alignment construction (`alignment.rs`)

**Spec**: `docs/alignment-struct-design.md`

```rust
pub struct Cell { pub old_pos: usize, pub new_pos: usize, pub content: String, pub is_gap: bool, pub gap_span: usize, pub cost: f64 }
pub struct AlignmentRow { pub pattern_id: u32, pub pattern_label: String, pub level: usize, pub cells: Vec<Cell>, pub fully_matched: bool }
pub struct Alignment { pub new_symbols: Vec<String>, pub rows: Vec<AlignmentRow>, pub covered: Vec<bool>, pub e_cost: f64, pub level_costs: Vec<f64> }

pub fn build_alignment(raw: &RawAlignment, new_names: &[&str], old_patterns: &[&Pattern], grammar: &Grammar) -> Alignment
```

**Why `old_patterns: &[&Pattern]`**: `match_log` uses `old_idx` as an index into the slice passed to `beam_search`, not a pattern ID. `build_alignment` needs the same slice to resolve symbol names and gap constraints. The caller passes the same `&[&Pattern]` to both `beam_search` and `build_alignment`.

`build_alignment` steps:
1. Sort `match_log` by `(old_idx, new_pos)`.
2. Group by `old_idx` → one `AlignmentRow` per old pattern. Resolve via `old_patterns[old_idx]`:
   - `SymbolRef::Atom(_)` → `new_names[new_pos]`
   - `SymbolRef::Pattern(id)` → `format!("P{id}")`
   - If pattern has non-empty `gaps`: after building cells, insert synthetic gap `Cell` between adjacent cells where `cells[i+1].new_pos > cells[i].new_pos + 1`. Gap cell: `is_gap=true`, `content=format!("<{}>", gap_span)`, `gap_span=next.new_pos - prev.new_pos - 1`, `cost=0.0`.
3. `fully_matched = non-gap cell count == pattern.symbols.len()`
4. Sort rows by `(level, first_new_pos().unwrap_or(usize::MAX))` ascending.
5. `level_costs`: `Vec::new()` — Phase 1d fills this in.

`impl fmt::Display for Alignment`: classic SP table format. See spec for column/row rules.

`impl Alignment { pub fn unmatched_symbols(&self) -> Vec<&str> }` — derive from `covered`, no stored field.

Done when: `build_alignment` round-trips a known beam result into the correct table string.

---

## Phase 1d — Training loop (`engine.rs`)

**Spec**: `docs/grammar-spec.md` implementation order steps 4-5, current `src_old/engine.rs` for reference on MDL gate and n-gram cold-start logic.

Key changes from old code:
- `SpmaEngine` operates on `Grammar` directly, not parallel `old_patterns`/`new_patterns` vecs.
- `extract_frequent_ngrams` and `extract_frequent_ngrams_ids` merged into one function — the duplicate (F8) is gone.
- `build_next_level_patterns` pid ordering uses `match_log` first-match position, not the broken `contains` scan (F1 fix).
- `extract_learned_patterns` takes `&Pattern` with `Vec<SymbolRef>`, not `Vec<Symbol>`.
- N-level outer loop populates `Grammar::levels`.
- `symbol_to_old` in `beam_search` indexes both `SymbolRef::Atom(id)` and `SymbolRef::Pattern(id)` by their inner `u32` — already fixed. At level 1, `new: &[u32]` contains pattern IDs; `SymbolRef::Pattern(id)` matches when `new[p] == id`. No further change needed.

`InferResult` (authoritative, in `lib.rs`):

```rust
pub struct InferResult {
    pub e_cost: f64,
    pub is_anomaly: bool,
    pub cd: f64,
    pub e_norm: f64,
    pub anomaly_percentile: f64,
    pub level_costs: Vec<f64>,
    pub level_e_norms: Vec<f64>,
    pub alignment: Alignment,
}
```

No `unmatched` field, no `alignment: String`, no `level_alignments: Vec<String>`.

Done when: `Spma::train` + `Spma::infer` work end-to-end on a small corpus. Alignment table prints correctly.

---

## Phase 1e — Calibrated score (`engine.rs`)

**Spec**: `docs/calibrated-score-design.md`

- `infer_internal(seq: &[u32]) -> (f64, f64)` — returns `(e_cost, raw_new_cost)`. No match log, no `AlignmentRow` allocation. Used during training to populate `EDistribution`.
- **Cost model upgrade**: switch from uniform `log2(n_atoms)` to frequency-based `-log2(freq/total)` per atom. Compute atom frequencies over corpus before building costs vec. Update `train` and `infer` to use per-symbol costs. `infer_internal` must use the same cost table. This is the correct baseline for all MDL, E_norm, and anomaly_percentile computations.
- `E_norm = e_cost / raw_new_cost`. Guard: `if raw_new_cost < 1e-12 { skip }`.
- Per-level denominator: `level_e_norms[i] = level_costs[i] / raw_level_i_cost` where `raw_level_i_cost` is the raw cost of the pid sequence at level i — NOT the original atom sequence cost.
- `anomaly_percentile`: binary search via `partition_point` into `sorted_e_norms`.
- `is_anomaly`: `e_norm > grammar.e_distribution.threshold`. Default threshold `0.0` — identical to old `e_cost > 0.0` behavior.

Done when: `InferResult` has correct `e_norm`, `anomaly_percentile`, `level_e_norms`. Unit test: train on 10 identical sequences, verify all training sequences have `e_norm = 0.0` and `anomaly_percentile = 0.0`.

---

## Phase 2a — Gap matching (`beam.rs`, `engine.rs`)

**Spec**: `docs/gap-patterns-design.md`

### Beam change — `can_extend` with gap constraint

```rust
fn can_extend(&self, old_idx: usize, old_pos: usize, new_pos: usize, patterns: &[&Pattern]) -> bool {
    let pat = patterns[old_idx];
    match self.new_cursors.get(&old_idx) {
        None => old_pos == 0 && new_pos >= self.max_covered_new,
        Some(&prev_new) => {
            if pat.gaps.is_empty() || old_pos == 0 {
                new_pos == prev_new + 1
            } else {
                let gap = &pat.gaps[old_pos - 1];
                let skip = new_pos.saturating_sub(prev_new + 1);
                skip >= gap.min && skip <= gap.max
            }
        }
    }
}
```

No new `PartialAlignment` fields. Gap-interior positions are uncovered — not in `match_log`, contribute to E. `build_alignment` infers gap cells from distance between adjacent match events for the same pattern.

### Grammar induction change — `extract_learned_patterns`

Merge adjacent covered spans separated by `<= max_gap` uncovered positions into a single gap pattern instead of two separate contiguous patterns. `max_gap` is a field on `SpmaEngine`, default 3.

### N-gram cold-start change

Count co-occurring pairs within a bounded window in addition to contiguous bigrams/trigrams. Candidates are `(Vec<SymbolRef>, Vec<GapConstraint>, u32)`.

Done when: test corpus `10× [TRIP, X, RESTORATION]` (X varies) learns `Pattern{symbols:[TRIP,RESTORATION], gaps:[{0,1}]}` and infers `[TRIP, Y, RESTORATION]` with `e_norm < 1.0`.

---

## Phase 3 — Serialization (deferred)

**Do not implement until**: model is stable AND there is at least one real grammar file a user cannot afford to break.

When the time comes: evaluate `postcard` first (compact, no-schema, pure Rust). Fall back to `serde_json` for human-readable debugging. No magic bytes, no CRC32, no migration tables until version 2 actually exists.

---

## What is never implemented

- `SymbolRef::Gap` — gaps are `Pattern::gaps: Vec<GapConstraint>`, not a symbol variant
- `SymbolRef::Var` — not needed
- IC/OC (`SymbolRole`) — `SymbolRef::Pattern(pid)` handles hierarchical composition explicitly
- `CellContent` enum — `Cell { is_gap: bool, content: String }` covers all cases
- `AlignmentType` (FullA/FullB/FullC/Partial) — dead classification, never used downstream
- `Symbol` wrapper struct — patterns store `Vec<SymbolRef>` directly
- `SymbolType`, `SymbolStatus` — gone with `Symbol`
- Default anomaly threshold `p90` — always `0.0`

---

## Dependency summary

```
Phase 0  →  Phase 1a  →  Phase 1b  →  Phase 1c  →  Phase 1d  →  Phase 1e  →  Phase 2a  →  Phase 3
scaffold    model        beam          alignment     engine        calibration   gaps          serialization
```

Each phase has no circular dependency on the next. Phases 1b and 1c can be developed and tested independently of 1d. Phase 2a is a contained change to `beam.rs` and `engine.rs` only — no struct changes to `Alignment` or `InferResult`.
