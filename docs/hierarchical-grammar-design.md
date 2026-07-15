# Hierarchical Grammar — Fix B Design

Issue #5 Fix B: grammar-level sequence patterns (N-level SP theory).

This document is a concrete implementation design, not a conceptual overview.
The conceptual rationale is in [known-issues.md](known-issues.md) Issue #5.

## Problem statement

The current engine is level-1 only: atomic symbols → Old patterns → New coverage.
Inter-pattern ordering is not represented in the grammar. Two sequences
`[A, B, C, D]` and `[C, D, A, B]` both score E=0 if the grammar contains
`[A,B]` and `[C,D]`, because the beam assigns patterns left-to-right and both
orderings are left-to-right consistent.

Fix B: add a level-2 grammar layer whose atoms are level-1 pattern IDs.
Order violations become coverage failures at level 2 — same mechanism, same beam, same MDL.

## Core data model change

### Current

```
Symbol { name: u32 }    // u32 = atomic symbol ID from Interner
Pattern { symbols: Vec<Symbol>, pattern_id: u32 }
SpmaEngine { old_patterns: Vec<Pattern> }
```

### Required

```rust
// src/model.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolRef {
    Atom(u32),      // atomic symbol — existing path, unchanged
    Pattern(u32),   // reference to a Pattern by pattern_id
}

pub struct Symbol {
    pub name: SymbolRef,   // was: u32
    // all other fields unchanged
}
```

`Symbol::new(id: u32)` keeps its current signature — callers unchanged. Add:

```rust
impl Symbol {
    pub fn new_pattern_ref(pattern_id: u32) -> Self {
        Self { name: SymbolRef::Pattern(pattern_id), ..Self::default() }
    }

    pub fn atom_id(&self) -> Option<u32> {
        match self.name { SymbolRef::Atom(id) => Some(id), _ => None }
    }

    pub fn pattern_id(&self) -> Option<u32> {
        match self.name { SymbolRef::Pattern(id) => Some(id), _ => None }
    }
}
```

Existing `Symbol::new(id)` becomes `Symbol { name: SymbolRef::Atom(id), .. }`.
All existing callers that pass `u32` directly still compile.
`Symbol::matches` compares `name` enum values — unchanged semantics.

### Grammar store

Replace the hardcoded level-2 fields with a `Vec` of levels. Level 0 = existing
`old_patterns` (atoms). Level 1+ = pattern-ref sequences. The same struct holds all levels.

```rust
/// One grammar level: the Old patterns and their corpus costs.
pub struct GrammarLevel {
    pub old_patterns: Vec<Pattern>,   // level 0: atoms; level N>0: SymbolRef::Pattern refs
    pub corpus_costs: Vec<f64>,       // Shannon costs indexed by symbol/pattern ID
}

pub struct SpmaEngine {
    // level-0 kept as-is for backward compat with all existing code paths
    pub old_patterns: Vec<Pattern>,
    pub new_patterns: Vec<Pattern>,
    pub corpus_costs: Vec<f64>,

    // higher levels — Vec grows as each level produces a non-empty grammar
    pub grammar_levels: Vec<GrammarLevel>,  // index 0 = level 1, index 1 = level 2, ...

    // shared — unchanged
    pub interner: Interner,
    pub symbol_frequencies: HashMap<u32, u32>,
    pub original_alphabet: HashSet<u32>,
    pub next_pattern_id: u32,
    pub verbose: bool,
    pub max_cycles: u32,
    pub keep_rows: u32,
    pub max_levels_safety_cap: u8,  // NEW: emergency circuit-breaker only (default: 16)
}
```

`grammar_levels` grows by one entry each time a level produces a non-empty grammar.
A corpus that only has level-1 structure ends up with `grammar_levels = []`.
A corpus with level-1 and level-2 structure ends up with `grammar_levels = [GrammarLevel_l2]`.
Level-3 structure adds a second entry, and so on up to `max_levels`.

## Beam search changes

### Current beam input

```rust
beam_search(new: &[u32], old: &[Vec<u32>], beam_k: usize, costs: &[f64])
```

### Required generalisation

The beam operates on symbol IDs, not on `SymbolRef` — keep it that way. The
level-2 beam reuses the exact same `beam_search` function with a different ID
namespace:

- Level-1 beam: `new = atom IDs`, `old = Vec<Vec<atom_id>>`, `costs = atom_costs`
- Level-2 beam: `new = pattern_id sequence`, `old = Vec<Vec<pattern_id>>`, `costs = pattern_costs`

Both fit the existing `beam_search(new: &[u32], old: &[Vec<u32>], ...)` signature.
No changes to `beam.rs`.

The only requirement: pattern IDs used at level 2 must be stable (not reassigned
after level-1 convergence). `pattern_id` is already assigned at insertion and
never changed — this holds.

## Learning pipeline changes

### Current `learn()` flow

```
1. cold-start n-gram miner → old_patterns
2. loop until convergence:
     beam search over new_patterns vs old_patterns
     MDL gate → add/reject candidates
3. snapshot corpus_costs
```

### New `learn()` flow

```
1. [unchanged] cold-start n-gram miner → old_patterns
2. [unchanged] loop until convergence: beam + MDL → old_patterns
3. [unchanged] snapshot corpus_costs (atom level)
4. [NEW] loop up to max_levels:
     a. build next-level New sequences (coverage sequences from current level)
     b. if next-level New sequences are all empty or too short → stop
     c. run learn_one_level() → get new GrammarLevel
     d. if resulting grammar is empty → stop (no repeated structure at this level)
     e. push GrammarLevel to grammar_levels
     f. continue to next level using this level's patterns as input
```

### Core helper — `build_next_level_patterns`

Reusable at every level. Input: current-level New patterns + current-level Old patterns + costs.
Output: next-level New patterns (each is a sequence of current-level pattern IDs).

```rust
fn build_next_level_patterns(
    new_pats: &[Pattern],
    old_pats: &[Pattern],
    keep_rows: usize,
    costs: &[f64],
    next_id: &mut u32,
) -> Vec<Pattern> {
    // project old patterns to their ID sequences (atoms at l0, pattern refs at l1+)
    let old_id_vecs: Vec<Vec<u32>> = old_pats.iter()
        .map(|p| p.symbols.iter().map(|s| match s.name {
            SymbolRef::Atom(id) => id,
            SymbolRef::Pattern(id) => id,
        }).collect())
        .collect();

    new_pats.iter().filter_map(|np| {
        let ids: Vec<u32> = np.symbols.iter().map(|s| match s.name {
            SymbolRef::Atom(id) => id,
            SymbolRef::Pattern(id) => id,
        }).collect();

        let best = beam_search(&ids, &old_id_vecs, keep_rows, costs)
            .into_iter().next()?;

        // Extract used pattern IDs ordered by first covered New position
        let mut pid_starts: Vec<(u32, usize)> = best.old_pattern_indices.iter()
            .filter_map(|&oi| {
                let pid = old_pats[oi].pattern_id;
                let start = ids.iter().enumerate()
                    .find(|&(i, _)| best.covered_new[i]
                        && old_id_vecs[oi].contains(&ids[i]))
                    .map(|(i, _)| i)?;
                Some((pid, start))
            })
            .collect();
        pid_starts.sort_by_key(|&(_, s)| s);
        let pid_seq: Vec<u32> = pid_starts.into_iter().map(|(pid, _)| pid).collect();

        if pid_seq.is_empty() { return None; }

        let symbols: Vec<Symbol> = pid_seq.iter()
            .map(|&pid| Symbol::new_pattern_ref(pid))
            .collect();
        let mut pat = Pattern::new(symbols, *next_id);
        *next_id += 1;
        Some(pat)
    }).collect()
}
```

### Core helper — `learn_one_level`

Extracted from the existing `learn()` body. Takes New patterns, returns converged
Old patterns + corpus costs. Beam, MDL gate, n-gram miner, convergence check
are all identical to level-0.

```rust
fn learn_one_level(
    &mut self,
    new_pats: Vec<Pattern>,
) -> Result<(Vec<Pattern>, Vec<f64>)>
// returns (old_patterns_for_this_level, corpus_costs_for_this_level)
```

### `learn()` outer loop (N levels)

```rust
// After level-0 convergence + corpus_costs snapshot:

let mut current_new = self.new_patterns.clone();
let mut current_old = self.old_patterns.clone();
let mut current_costs = self.corpus_costs.clone();

loop {
    // Safety circuit-breaker — not the expected termination path.
    if self.grammar_levels.len() >= self.max_levels_safety_cap as usize { break; }

    let next_new = build_next_level_patterns(
        &current_new, &current_old, self.keep_rows as usize,
        &current_costs, &mut self.next_pattern_id,
    );

    // Natural termination: nothing to compress at this level.
    let viable = next_new.iter().filter(|p| p.symbols.len() >= 2).count();
    if viable == 0 { break; }

    let (next_old, next_costs) = self.learn_one_level(next_new.clone())?;

    // Natural termination: MDL gate rejected everything — no repeated structure.
    if next_old.is_empty() { break; }

    self.grammar_levels.push(GrammarLevel {
        old_patterns: next_old.clone(),
        corpus_costs: next_costs.clone(),
    });

    current_new = next_new;
    current_old = next_old;
    current_costs = next_costs;
}
```

**Termination argument (SP theory):** Each level's "corpus" is the sequence of pattern IDs
from the level below. Alphabet size = `current_old.len()`. At each level, the alphabet can
only shrink or stay the same (patterns that recur get grouped; singletons drop out via MDL
gate). When the alphabet collapses to one symbol, no bigram is possible →
`build_next_level_patterns()` returns sequences of length ≤ 1 → `viable == 0` → stops.
In practice: corpus sparsity kills higher levels well before alphabet collapse.

**`max_levels_safety_cap` (default 16):** Emergency guard against adversarial corpora or
implementation bugs. Not a design parameter. `set_max_levels()` is renamed
`set_max_levels_safety_cap()` to make the semantics explicit. Expected termination is always
the `viable == 0` or `next_old.is_empty()` path.

## Inference changes

### Current `infer()` flow

```
1. resolve atom IDs
2. identify unknowns
3. build costs[] from old_patterns + corpus_costs fallback
4. beam_search(atom_ids, old_patterns) → BeamAlignment
5. compute E, CD
```

### New `infer()` flow

```
1. [unchanged] resolve atom IDs, identify unknowns
2. [unchanged] level-0 beam → BeamAlignment; compute e_cost_l0, coverage

3. [NEW] for each GrammarLevel in grammar_levels:
     a. extract pid_sequence from previous level's alignment
        (same logic as build_next_level_patterns, but for a single sequence)
     b. if pid_sequence is empty → stop, no further levels contribute
     c. build costs[] from this level's old_patterns + corpus_costs fallback
     d. beam_search(pid_sequence, this_level_old_patterns) → alignment
     e. compute e_cost for this level (uncovered pid positions)
     f. push e_cost and alignment string to per_level results

4. total_e = e_cost_l0 + sum(per_level e_costs)
```

This loop is bounded by `grammar_levels.len()` — naturally zero iterations if no
higher-level grammar was learned. Inference cost grows linearly with depth.

### `InferResult` additions

```rust
pub struct InferResult {
    // existing fields unchanged
    pub e_cost: f64,          // total: sum of all levels
    pub cd: f64,
    pub is_anomaly: bool,
    pub unmatched: Vec<String>,
    pub alignment: String,    // level-0 alignment (unchanged)

    // new — one entry per grammar level beyond 0
    pub level_costs: Vec<f64>,       // e_cost per level: [l1, l2, l3, ...]
    pub level_alignments: Vec<String>, // alignment table per level
}
```

`e_cost` = `e_cost_l0 + level_costs.iter().sum()` — backward-compatible sum.
`is_anomaly` = `e_cost > 0` — unchanged.
`level_costs.is_empty()` when only level-0 grammar exists (current behaviour, no regression).

Callers that only check `e_cost` and `is_anomaly` need no changes.
Callers that want to know *which level* the anomaly comes from inspect `level_costs[i]`.

### Level-2 alignment table

`write_alignment_table` currently takes `&Interner` to resolve `u32 → &str`.
For level-2, names are pattern IDs, not atom IDs. Two options:

**A (simpler):** Build a second `Interner`-like structure that maps `pattern_id → display_name`
where `display_name` is the joined atom names of that level-1 pattern:
`"[A B]"`, `"[BREAKER_OPEN UNDERVOLTAGE]"`. Pass it to `write_alignment_table`
unchanged — function signature unchanged.

**B (cleaner):** Generalise `write_alignment_table` to accept `&dyn Fn(u32) -> String`
resolver instead of `&Interner`. Level-1 uses `|id| interner.name(id)`, level-2 uses
`|pid| format_pattern_name(pid, &old_patterns, &interner)`.

Option A has zero API surface change. Option B is the right abstraction if
level-3+ is ever added. Recommend A for the initial implementation.

## Persistence changes

`GrammarSnapshot` gains one field replacing the two hardcoded l2 variants:

```rust
#[derive(Serialize, Deserialize)]
struct GrammarLevelSnapshot {
    old_patterns: Vec<Pattern>,
    corpus_costs: Vec<f64>,
}

struct GrammarSnapshot {
    format_version: u8,                     // new: 1 = current, 2 = hierarchical
    old_patterns: Vec<Pattern>,             // existing (level-0)
    interner_names: Vec<String>,            // existing
    corpus_costs: Vec<f64>,                 // existing (level-0)
    grammar_levels: Vec<GrammarLevelSnapshot>,  // new: empty for v1 files
}
```

`Symbol` serialization must handle `SymbolRef`. Since `SymbolRef` is a serde enum,
bincode handles it natively — no custom serializer needed. File format not
backward-compatible (wire type of `Symbol::name` changes from `u32` to enum tag + u32).

`load()` detects `format_version = 1` → fills `grammar_levels = vec![]`, restores
`Symbol::name` as `SymbolRef::Atom(id)`. Old files load correctly, without higher levels.

## API surface

No breaking changes to `Spma` public API. Additions only:

```rust
impl Spma {
    // existing
    pub fn grammar_size(&self) -> usize { ... }      // level-0 only, unchanged

    // new
    pub fn grammar_depth(&self) -> usize {
        // number of levels with non-empty grammar (1 = level-0 only, 2 = l0+l1, ...)
        1 + self.inner.grammar_levels.len()
    }
    pub fn grammar_size_at(&self, level: usize) -> usize {
        // level 0 = old_patterns; level N>0 = grammar_levels[N-1]
        ...
    }
    // emergency cap — not a design parameter; expected termination is MDL-driven
    pub fn set_max_levels_safety_cap(&mut self, n: u8) { self.inner.max_levels_safety_cap = n; }
}
```

`set_max_cycles` applies to every level (same field used by `learn_one_level`).

## What this resolves

```
New = [C, D, A, B]
Training corpus = [[A,B,C,D]]×10

Level-1: grammar learns [A,B] and [C,D].
  [C,D] covers new[0..1], [A,B] covers new[2..3].
  e_cost_l1 = 0.0

Level-2: new_patterns_l2 from training = [[pid(A,B), pid(C,D)]]×10
  n-gram miner: bigram [pid(A,B), pid(C,D)] appears 10× → enters grammar.
  Infer pid_sequence = [pid(C,D), pid(A,B)] (reversed).
  Level-2 beam: [pid(C,D), pid(A,B)] vs [[pid(A,B), pid(C,D)]].
  pid(C,D) at position 0 — no level-2 pattern starts with pid(C,D) → uncovered.
  e_cost_l2 > 0. is_anomaly = true. ✓
```

## What this does NOT resolve

- Level-3+ requires another recursive pass (same algorithm, pattern-of-pattern IDs).
  The design supports it but level-2 is the priority.
- Sequences with no level-1 coverage produce empty level-2 input — level-2 E=0
  even when anomalous (all-unknown sequences). Correct: level-1 already fires E>0
  in that case.
- Short corpora may not produce stable level-2 patterns (MDL gate rejects if
  pattern-ID bigrams appear only once). The result is `old_patterns_l2 = []` and
  `e_cost_l2 = 0` always — graceful degradation to current behaviour.

## Implementation order

1. `src/model.rs`:
   - Add `SymbolRef` enum.
   - Change `Symbol::name: u32` → `Symbol::name: SymbolRef`.
   - Update `Symbol::new(id: u32)` to wrap in `SymbolRef::Atom(id)` — all callers unchanged.
   - Add `Symbol::new_pattern_ref(pid: u32)`.
   - Add `Symbol::atom_id() -> Option<u32>` and `Symbol::pattern_id() -> Option<u32>`.

2. `src/intern.rs`: no changes. Higher levels use pattern IDs directly, not interned strings.

3. `src/beam.rs`: no changes. Already operates on `&[u32]` — pattern IDs fit unchanged.

4. `src/engine.rs`:
   - Add `GrammarLevel` struct.
   - Add `grammar_levels: Vec<GrammarLevel>` and `max_levels: u8` to `SpmaEngine`.
   - Extract `learn_one_level()` from existing `learn()` body.
   - Add `build_next_level_patterns()`.
   - After level-0 convergence, run the N-level outer loop.

5. `src/lib.rs`:
   - `GrammarSnapshot`: add `format_version`, `grammar_levels: Vec<GrammarLevelSnapshot>`.
   - `load()`: handle version 1 (fill `grammar_levels = vec![]`).
   - `InferResult`: add `level_costs: Vec<f64>`, `level_alignments: Vec<String>`.
   - `infer()`: add N-level inference loop; sum into `e_cost`.
   - `Spma`: add `grammar_depth()`, `grammar_size_at()`, `set_max_levels()`.

6. `tests/`: add tests (see below).

## Tests to add

```
// tests/hierarchical.rs

// ── Level-2 ───────────────────────────────────────────────────────────────

fn l2_grammar_forms_after_repeated_ordered_corpus()
  Train [[A,B,C,D]]×10.
  Assert grammar_depth() >= 2 (level-2 grammar formed).

fn l2_correct_order_zero_level_cost()
  Train [[A,B,C,D]]×10.
  Infer [A,B,C,D] → level_costs[0] == 0.0 (level-1 E=0).

fn l2_reversed_order_nonzero_level_cost()
  Train [[A,B,C,D]]×10.
  Infer [C,D,A,B] → level_costs[0] > 0.0, is_anomaly == true.

fn l2_varied_order_corpus_no_l2_grammar()
  Train [[A,B,C,D]]×5 + [[C,D,A,B]]×5.
  grammar_depth() == 1 (no dominant pattern-ID bigram → no level-2 grammar).
  Infer [C,D,A,B] → level_costs is empty, e_cost == e_cost_l0.

fn l2_save_load_roundtrip_preserves_level_costs()
  Train [[A,B,C,D]]×10, save, load.
  Infer [C,D,A,B] before and after → level_costs identical.

fn l2_all_unknown_sequence_does_not_fire_l2()
  Train [[A,B,C,D]]×10.
  Infer [X,Y,Z,W] → e_cost > 0 (unknown), level_costs.is_empty() or level_costs[0]==0
  (empty pid_sequence → level-2 skipped).

// ── Level-3 ───────────────────────────────────────────────────────────────
// Level-3 requires episodes whose ordering recurs: e.g. a corpus of episodes
// where each episode is itself an ordered sequence of ordered sub-sequences.

fn l3_grammar_forms_on_episode_corpus()
  // Episode = two sub-sequences always in same order.
  // Corpus = 10 episodes, each = [A,B,C,D,E,F,G,H] where [A,B],[C,D] always precede [E,F],[G,H].
  // Level-0: learns [A,B],[C,D],[E,F],[G,H].
  // Level-1: learns [[AB],[CD],[EF],[GH]] as ordered sequence.
  // Level-2 needs repeated level-1 patterns across episodes → may form if varied episodes.
  // Minimum: grammar_depth() >= 2 after training on 10 identical episodes.
  // (level-3 only if episodes themselves recur in pairs)
  Train 10× [A,B,C,D,E,F,G,H].
  Assert grammar_depth() >= 2.

fn loop_terminates_naturally_on_flat_corpus()
  // Corpus with no hierarchical structure: unique sequences, no pattern recurrence.
  Train [[A,B,C,D],[E,F,G,H],[I,J,K,L]] (all distinct symbols).
  Assert grammar_depth() == 1 (MDL gate kills everything; no safety cap needed).

fn safety_cap_blocks_deep_hierarchy_when_set_low()
  engine.set_max_levels_safety_cap(1).
  Train [[A,B,C,D]]×10.
  Assert grammar_depth() <= 2 (cap enforced; not a theory assertion — just a guard test).

// ── Depth-independent ─────────────────────────────────────────────────────

fn grammar_size_at_returns_correct_counts()
  Train [[A,B,C,D]]×10.
  grammar_size_at(0) == grammar_size() (level-0, unchanged).
  grammar_size_at(1) > 0 if grammar_depth() >= 2.
  grammar_size_at(99) == 0 (level doesn't exist).

fn e_cost_equals_sum_of_level_costs()
  Train [[A,B,C,D]]×10.
  Infer [C,D,A,B].
  Assert (e_cost - e_cost_l0 - level_costs.iter().sum()).abs() < 1e-9.
```
