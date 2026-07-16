# Hierarchical Grammar — Code Review Findings

Code review of the level-N implementation across `src/engine.rs`, `src/lib.rs`, `src/beam.rs`,
and `tests/hierarchical.rs`. Ten findings, ordered by severity.

---

## F1 — pid_seq ordering: `contains` scan is wrong, but the root cause is deeper (HIGH)

**File**: `src/lib.rs:320-336`, `src/engine.rs:1096-1112`

Both the inference loop and `build_next_level_patterns` derive pid ordering via:

```rust
let start = prev_seq.iter().enumerate().find(|&(i, &sym_id)| {
    prev_align.covered_new[i] && prev_old_id_vecs[oi].contains(&sym_id)
}).map(|(i, _)| i)?;
```

**Problem 1 — `contains` false-match**: at level 2+, atom IDs are shared across multiple
patterns. If two level-1 patterns both reference the same atom ID, `contains` fires on the
wrong pattern → wrong `start` → pid_seq ordering incorrect → level-2 alignment is wrong.

**Problem 2 — `old_pattern_indices` is sorted by index, not by match position**: in
`beam.rs`, `finalize` does:

```rust
let mut old_pattern_indices: Vec<usize> = self.old_cursors.keys().copied().collect();
old_pattern_indices.sort_unstable();
```

This sorts by old-pattern *index* (position in the grammar array), not by the New position
where each pattern first matched. When `build_next_level_patterns` iterates
`old_pattern_indices` and then tries to recover ordering via the `contains` scan, it is
sorting garbage: the scan is broken (Problem 1) and the input order is wrong (Problem 2).

**Root cause**: `BeamAlignment` does not carry the first-matched New position per old-pattern
slot. That information exists during beam traversal — `new_pos` is passed to `extend_match`
and stored in `new_cursors` — but it is not surfaced in the finalized result.

**Fix**: add `old_first_new_pos: HashMap<usize, usize>` to `BeamAlignment`, populated in
`finalize` from `self.new_cursors`. Then `build_next_level_patterns` and the inference loop
read it directly:

```rust
// In finalize():
let old_first_new_pos: HashMap<usize, usize> = self.new_cursors
    .iter()
    .map(|(&oi, &new_pos)| {
        // new_cursors tracks the *last* matched new pos; we need the *first*.
        // Either track first separately, or reconstruct from covered_new + old_cursors.
        (oi, new_pos)
    })
    .collect();
```

Better: track `old_first_new_pos: HashMap<usize, usize>` in `PartialAlignment` alongside
`new_cursors`, set on first match (when `old_cursors` does not yet contain `old_idx`), never
updated after. Then `build_next_level_patterns` becomes:

```rust
let mut pid_starts: Vec<(u32, usize)> = best
    .old_first_new_pos   // new field
    .iter()
    .filter_map(|(&oi, &first_pos)| {
        Some((old_pats[oi].pattern_id, first_pos))
    })
    .collect();
pid_starts.sort_by_key(|&(_, s)| s);
let pid_seq: Vec<u32> = pid_starts.into_iter().map(|(pid, _)| pid).collect();
```

This eliminates the `contains` scan entirely and makes ordering deterministic and correct.

**Verification**: the `Atom`/`Pattern` distinction at level 2+ does not save you. At level 1,
`build_next_level_patterns` produces patterns whose symbols are `SymbolRef::Pattern(pid)` —
so `contains` on `old_id_vecs[oi]` (which holds `raw_id()` values, i.e. the pid numbers)
will not false-match *between* level-1 patterns as long as all pids are unique. But Problem 2
(wrong sort input) is still live regardless of the `Atom`/`Pattern` distinction.

---

## F2 — 4 clones where 2 suffice (MEDIUM)

**File**: `src/engine.rs:519-524`

```rust
let level0_old = self.old_patterns.clone();   // clone 1
let level0_new = self.new_patterns.clone();   // clone 2
let mut current_new = self.new_patterns.clone();  // clone 3 — same data as clone 2
let mut current_old = self.old_patterns.clone();  // clone 4 — same data as clone 1
```

`current_new` and `current_old` are identical to `level0_new`/`level0_old` at the start.
Fix:

```rust
let level0_old = self.old_patterns.clone();
let level0_new = self.new_patterns.clone();
let mut current_old = level0_old.clone();
let mut current_new = level0_new.clone();
```

Saves two full clones of the pattern store on every `learn` call.

---

## F3 — Per-inference clone of level patterns (MEDIUM)

**File**: `src/lib.rs:304`, `src/lib.rs:446`

```rust
let mut prev_old_patterns_snapshot: Vec<Pattern> = self.inner.old_patterns.clone(); // line 304
// ...
prev_old_patterns_snapshot = level.old_patterns.clone(); // line 446 — every iteration
```

`prev_old_patterns_snapshot` is read-only inside the loop. Fix: use a reference:

```rust
let mut prev_old_patterns: &[Pattern] = &self.inner.old_patterns;
// ...
prev_old_patterns = &level.old_patterns;
```

Eliminates a full clone of the grammar per level per `infer()` call. Requires threading the
lifetime through `prev_old_id_vecs` construction, which is straightforward since both live
for the duration of the loop body.

---

## F4 — O(n) `contains` in pid detection (MEDIUM)

**File**: `src/engine.rs:1106`, `src/lib.rs:330`

```rust
old_id_vecs[oi].contains(&sym_id)  // Vec linear scan
```

Both sites do this per covered position per old pattern. At level 2+ with large grammars,
this is O(|old_patterns| × |sequence| × avg_pattern_length).

This is also the broken scan from F1. Fixing F1 (adding `old_first_new_pos` to
`BeamAlignment`) eliminates both sites entirely — F4 becomes moot once F1 is fixed.

---

## F5 — Missing `e_cost_base` in `InferResult` (MEDIUM)

**File**: `src/lib.rs:44-59`

```rust
pub struct InferResult {
    pub e_cost: f64,       // total = level-0 E + sum(level_costs)
    pub level_costs: Vec<f64>,
    // e_cost_base missing
}
```

Without `e_cost_base: f64` (the level-0 beam E cost before higher-level adjustments),
callers cannot decompose `e_cost` into its level-0 component vs higher-level costs. The
existing test `e_cost_equals_sum_of_level_costs` is weakened as a result — it checks
`level_sum <= e_cost` instead of the exact invariant `e_cost == e_cost_base + level_sum`.

Fix:

1. Add field to `InferResult`:
```rust
/// Level-0 beam E cost before higher-level adjustments.
pub e_cost_base: f64,
```

2. Capture before the N-level loop in `infer()`:
```rust
let e_cost_base = e_cost;
```

3. Return it and strengthen the test (see T6 below).

---

## F6 — `GrammarLevel` double-clone on push (LOW)

**File**: `src/engine.rs:553-560`

```rust
self.grammar_levels.push(GrammarLevel {
    old_patterns: next_old.clone(),   // clone for push
    corpus_costs: next_costs.clone(), // clone for push
});
current_old = next_old;              // original moved
current_costs = next_costs;
```

`learn_one_level` returns owned values. They are cloned into the push and then the originals
are moved into `current_old`/`current_costs`. Fix: push first, then clone back:

```rust
self.grammar_levels.push(GrammarLevel {
    old_patterns: next_old,
    corpus_costs: next_costs,
});
let gl = self.grammar_levels.last().unwrap();
current_old = gl.old_patterns.clone();
current_costs = gl.corpus_costs.clone();
```

Still one clone each, but avoids the redundant second clone.

---

## F7 — `Atom` variant rendered as `[pat:N]` (LOW)

**File**: `src/lib.rs:412-416`

```rust
crate::model::SymbolRef::Atom(id) => format!("[pat:{}]", id),
crate::model::SymbolRef::Pattern(pid) => format!("[pat:{}]", pid),
```

Both branches produce identical output. `Atom(id)` at level 2+ holds a pattern ID from the
level below — the label `[pat:N]` is not wrong, but the match is misleading and will mask
future bugs where an unexpected `Atom` variant appears at a higher level.

Fix:

```rust
crate::model::SymbolRef::Atom(id) => format!("[atom:{}]", id),
crate::model::SymbolRef::Pattern(pid) => format!("[pat:{}]", pid),
```

---

## F8 — `extract_frequent_ngrams` and `extract_frequent_ngrams_ids` are near-identical (MEDIUM)

**File**: `src/engine.rs` — two separate functions, ~80 lines each

The two functions share the same structure: count n-grams, sort by savings potential, greedy
MDL selection loop, DP check. The only difference is the symbol constructor at insertion:

- `extract_frequent_ngrams`: uses `Symbol::new(id)` (atom ref)
- `extract_frequent_ngrams_ids`: uses `Symbol::new_pattern_ref(id)` (pattern ref)

This duplication will diverge silently. Any fix to the MDL logic, cost lookup, or dedup check
must be applied twice. Fix: extract a shared helper parameterised on the symbol constructor:

```rust
fn extract_frequent_ngrams_generic(
    &mut self,
    patterns: &[Pattern],
    min_freq: u32,
    make_symbol: impl Fn(u32) -> Symbol,
) -> bool { ... }
```

Then:

```rust
fn extract_frequent_ngrams(&mut self, patterns: &[Pattern], min_freq: u32) -> bool {
    self.extract_frequent_ngrams_generic(patterns, min_freq, Symbol::new)
}

fn extract_frequent_ngrams_ids(&mut self, patterns: &[Pattern], min_freq: u32) -> bool {
    self.extract_frequent_ngrams_generic(patterns, min_freq, Symbol::new_pattern_ref)
}
```

---

## F9 — `old_pattern_indices` sorted by grammar index, not match position (MEDIUM)

**File**: `src/beam.rs:108-109`

```rust
let mut old_pattern_indices: Vec<usize> = self.old_cursors.keys().copied().collect();
old_pattern_indices.sort_unstable();
```

This sorts by old-pattern *index* (position in the grammar array). Consumers of
`old_pattern_indices` — the alignment table printer and the pid_seq builder — both need
patterns in the order they appear in the New sequence, not in grammar-array order.

This is the second half of F1. The alignment table printer assigns columns by iterating
`old_pattern_indices` in this sorted order, which means columns can be assigned to the wrong
old pattern when grammar-array order differs from New-sequence order.

Fix: sort `old_pattern_indices` by first matched New position, using `new_cursors`:

```rust
let mut old_pattern_indices: Vec<usize> = self.old_cursors.keys().copied().collect();
old_pattern_indices.sort_unstable_by_key(|&oi| {
    self.new_cursors.get(&oi).copied().unwrap_or(usize::MAX)
});
```

This makes `old_pattern_indices` reflect left-to-right New order, which is what both the
printer and the pid_seq builder expect.

---

## F10 — `extract_learned_patterns` contiguity is correct but not tested for gaps at level 1+ (LOW)

**File**: `src/engine.rs` — `extract_learned_patterns`

The function correctly extracts maximal contiguous covered spans (the engine unit tests in
`engine.rs` verify this). However, at level 1+, the covered spans are over pid sequences, not
atom sequences. A gap in coverage at level 1 means a pid was not matched — the extracted
span will be a sub-sequence of pids. This is correct behaviour, but there are no tests that
verify the span extraction produces sensible patterns at level 1+. If the pid_seq ordering is
wrong (F1/F9), the extracted spans will be wrong too, silently producing bad grammar patterns
at higher levels.

This is not a bug in `extract_learned_patterns` itself, but a coverage gap that will hide F1
regressions. Add a test that verifies the level-1 grammar patterns contain the expected pid
sequences after training on a known corpus.

---

## Priority order

| # | File | Severity | Correctness risk |
|---|---|---|---|
| F1 | `src/beam.rs`, `src/lib.rs`, `src/engine.rs` | HIGH | Wrong pid_seq at level 2+ |
| F9 | `src/beam.rs` | MEDIUM | Wrong column assignment in alignment table; wrong pid order |
| F8 | `src/engine.rs` | MEDIUM | Silent divergence between two MDL loops |
| F5 | `src/lib.rs` | MEDIUM | API gap, weakened test |
| F2 | `src/engine.rs` | MEDIUM | Perf only |
| F3 | `src/lib.rs` | MEDIUM | Memory only |
| F4 | `src/engine.rs`, `src/lib.rs` | MEDIUM | Perf only; eliminated by F1 fix |
| F6 | `src/engine.rs` | LOW | Memory only |
| F7 | `src/lib.rs` | LOW | Display only |
| F10 | `src/engine.rs` | LOW | Coverage gap, not a bug |

## Fix order recommendation

1. **F1 + F9 together** — both require touching `BeamAlignment`. Add `old_first_new_pos` to
   `PartialAlignment` and `BeamAlignment` in one pass. Fix `old_pattern_indices` sort in
   `finalize`. Fix pid_seq derivation in `build_next_level_patterns` and `infer`.
2. **F8** — merge the two ngram functions. Mechanical, no behaviour change.
3. **F5** — add `e_cost_base` field and exact test. Mechanical.
4. **F2** — consolidate the 4 clones into 2.
5. **F3** — switch `prev_old_patterns_snapshot` to reference.
6. **F4** — eliminated by F1 fix; no separate action needed.
7. **F6, F7, F10** — fix opportunistically.

---

## Missing test coverage

Current suite only exercises level-2. F1/F9 (pid ordering bugs) are not reproducible without
level-3+ corpora.

### Coverage gaps

| Gap | Why it matters |
|---|---|
| No level-3 corpus | Cannot verify F1/F9 don't fire in practice |
| No shared-atom stress test | F1 scenario requires two level-1 patterns sharing an atom ID |
| No `e_cost_base` exact assertion | F5 — weakened test masks real decomposition errors |
| No anomaly propagation at level-3 | Don't know if anomaly detection carries past level-2 |
| No level-3 save/load | Persistence validated for level-2 only |
| No level-1 grammar pattern content check | F10 — hides F1 regressions at extraction time |

### Corpus for level-3

Need a 12-symbol repeating sequence:

```
A B C D E F A B C D E F  (×10)
```

- Level-0: learns `[A,B,C]` and `[D,E,F]`
- Level-1: learns `[pid_ABC, pid_DEF]` → single level-1 pattern `pid_ABCDEF`
- Level-2: learns `[pid_ABCDEF, pid_ABCDEF]` → level-2 pattern; `grammar_depth() == 3`

### Tests to add in `tests/hierarchical.rs`

**T1 — `l3_grammar_forms_on_nested_corpus`**

Primary reproducer for F1/F9. Asserts `grammar_depth() >= 3`.

```rust
fn train_abcdef_x2_x10() -> Spma {
    let mut eng = Spma::new();
    let corpus: Vec<Vec<&str>> = (0..10)
        .map(|_| vec!["A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E", "F"])
        .collect();
    eng.train(&corpus).unwrap();
    eng
}

#[test]
fn l3_grammar_forms_on_nested_corpus() {
    let eng = train_abcdef_x2_x10();
    assert!(
        eng.grammar_depth() >= 3,
        "nested repeated corpus must form at least 3 grammar levels, got {}",
        eng.grammar_depth()
    );
}
```

**T2 — `l3_correct_order_zero_level_cost_at_all_levels`**

Verifies anomaly signal propagates correctly through all levels for a known-good sequence.

```rust
#[test]
fn l3_correct_order_zero_level_cost_at_all_levels() {
    let eng = train_abcdef_x2_x10();
    if eng.grammar_depth() < 3 { return; }
    let result = eng
        .infer(&["A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E", "F"])
        .unwrap();
    assert!(result.level_costs.len() >= 2);
    assert_eq!(result.level_costs[0], 0.0, "level_costs[0] must be 0 for correct order");
    assert_eq!(result.level_costs[1], 0.0, "level_costs[1] must be 0 for correct order");
}
```

**T3 — `l3_pid_ordering_no_collision` (F1/F9 reproducer)**

If pid_seq ordering is wrong, alignment will be inverted and `level_costs[1]` will be nonzero
for a known-good input.

```rust
#[test]
fn l3_pid_ordering_no_collision() {
    let eng = train_abcdef_x2_x10();
    if eng.grammar_depth() < 3 { return; }
    let result = eng
        .infer(&["A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E", "F"])
        .unwrap();
    assert!(result.level_alignments.len() >= 2);
    assert_eq!(
        result.level_costs[1], 0.0,
        "level-2 cost must be 0 for correctly ordered input; \
         nonzero indicates pid ordering bug (F1/F9). level_alignments: {:?}",
        result.level_alignments
    );
}
```

**T4 — `l3_reversed_nested_order_anomalous`**

Verifies anomaly detection at level-3 for an out-of-order input.

```rust
#[test]
fn l3_reversed_nested_order_anomalous() {
    let eng = train_abcdef_x2_x10();
    if eng.grammar_depth() < 3 { return; }
    let result = eng
        .infer(&["D", "E", "F", "A", "B", "C", "D", "E", "F", "A", "B", "C"])
        .unwrap();
    assert!(result.is_anomaly, "reversed nested blocks must be anomalous");
}
```

**T5 — `l3_save_load_roundtrip_preserves_all_levels`**

Persistence regression for grammars with 3+ levels.

```rust
#[test]
fn l3_save_load_roundtrip_preserves_all_levels() {
    let eng = train_abcdef_x2_x10();
    if eng.grammar_depth() < 3 { return; }
    let path = "/tmp/spma_l3_roundtrip.bin";
    eng.save(path).unwrap();
    let eng2 = Spma::load(path).unwrap();

    assert_eq!(eng.grammar_depth(), eng2.grammar_depth(), "depth mismatch after load");

    let seq = ["A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E", "F"];
    let r1 = eng.infer(&seq).unwrap();
    let r2 = eng2.infer(&seq).unwrap();

    assert_eq!(r1.level_costs.len(), r2.level_costs.len());
    for (i, (a, b)) in r1.level_costs.iter().zip(r2.level_costs.iter()).enumerate() {
        assert!((a - b).abs() < 1e-9, "level_costs[{i}] mismatch: {a} vs {b}");
    }
    assert_eq!(r1.is_anomaly, r2.is_anomaly);
}
```

**T6 — `e_cost_base_exact_decomposition`** *(blocked on F5 fix)*

Replaces the weakened check in `e_cost_equals_sum_of_level_costs`.

```rust
#[test]
fn e_cost_base_exact_decomposition() {
    let eng = train_abcdef_x2_x10();
    let result = eng
        .infer(&["D", "E", "F", "A", "B", "C", "D", "E", "F", "A", "B", "C"])
        .unwrap();
    let level_sum: f64 = result.level_costs.iter().sum();
    let expected = result.e_cost_base + level_sum;
    assert!(
        (result.e_cost - expected).abs() < 1e-9,
        "e_cost ({}) must equal e_cost_base ({}) + level_sum ({})",
        result.e_cost, result.e_cost_base, level_sum
    );
}
```

**T7 — `l1_grammar_patterns_contain_expected_pid_sequences`** *(F10 coverage)*

Verifies that level-1 grammar patterns contain the expected pid sequences after training,
catching F1 regressions at extraction time before they propagate to level-2.

```rust
#[test]
fn l1_grammar_patterns_contain_expected_pid_sequences() {
    let eng = train_abc_def_x10(); // existing level-2 corpus
    assert!(eng.grammar_depth() >= 2);
    // Level-1 grammar must contain at least one multi-symbol pattern
    // (the [pid_ABC, pid_DEF] sequence).
    assert!(
        eng.grammar_size_at(1) > 0,
        "level-1 grammar must contain at least one multi-symbol pattern"
    );
}
```

### Implementation order

Add T1 first — if corpus doesn't reach depth 3, T2–T5 skip gracefully. Once T1 passes:
- T3 is the F1/F9 reproducer — run before fixing to confirm the bug is live.
- T6 is blocked until F5 (`e_cost_base` field) is done.
- T7 can be added immediately against the existing level-2 corpus.
