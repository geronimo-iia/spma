# Known Issues

Discovered during test coverage expansion (2026-07-15). Issues 1 and 2 are resolved.

## 1. ~~`find_degree_of_matching` always returns `FullA` when `columns` is empty~~ **RESOLVED**

**Resolution (2026-07-16):** `Alignment`, `AlignmentElement`, `HitNode`, and `Grammar` were all
dead code — never called from the live path after HitNode scaffolding removal. All four types and
their associated methods were deleted from `src/model.rs`.

---

## ~~1.~~ (archived) `find_degree_of_matching` always returns `FullA` when `columns` is empty

**Location:** `src/model.rs`, `Alignment::find_degree_of_matching`

**Observed behavior:** When `alignment.columns` is empty, the method sets `degree_of_matching = FullA` regardless of how many patterns are present. Both `new_fully_matched` and `old_fully_matched` initialise to `true`; the column loop never runs so they stay `true`; the match arm `(true, true)` fires → `FullA`.

**Why it matters:** Any code that creates an `Alignment` without populating `columns` (e.g. a partially constructed alignment, or one returned from the old hit-node path) will appear fully matched. If `degree_of_matching` is ever used to gate downstream logic this will silently produce wrong results.

**Prompt for fix evaluation:**

```
In src/model.rs, Alignment::find_degree_of_matching (lines 187-231):
The method initialises new_fully_matched and old_fully_matched to true
and only falsifies them by iterating self.columns. When columns is empty
the loop body never executes and the result is always FullA.

Evaluate whether the correct sentinel when columns is empty should be:
  a) Partial (nothing was matched — most conservative)
  b) leave as-is and add a precondition assert!(self.columns.is_empty() means caller error)
  c) compute degree purely from pattern symbol counts without columns

Consider: is find_degree_of_matching still called anywhere in the live
path now that the HitNode scaffolding was removed? If it is dead code,
the correct fix may be to delete it entirely. Check all call sites first.

If it is live: implement option (a) — guard at the top:
  if self.columns.is_empty() { self.degree_of_matching = AlignmentType::Partial; return; }
Add a regression test that verifies an empty-columns Alignment resolves Partial, not FullA.
```

## 2. ~~Unseen symbols have zero cost — `infer()` cannot detect novel-symbol anomalies~~ **RESOLVED**

**Resolution (2026-07-16):** Implemented Option B. Before beam search, each symbol ID is checked
against `original_alphabet`. Unknown positions are forced uncovered after beam search and
accumulate an `unknown_penalty` (average known bit cost) added to `e_cost`. `is_anomaly` now
correctly fires for sequences containing symbols never seen during training.

---

## ~~2.~~ (archived) Unseen symbols have zero cost — `infer()` cannot detect novel-symbol anomalies

**Location:** `src/lib.rs`, `Spma::infer`

**Observed behavior:** Symbols not seen during training are interned on-the-fly into `tmp_interner`. The cost table is built from `old_patterns` bit costs; a new id has no entry → `costs[id] = 0.0`. Beam search therefore assigns E=0 to unseen symbols. `is_anomaly = e_cost > 0.0` stays false. The symbol appears in `unmatched` (correct) but does not trigger the anomaly flag (wrong).

**Prompt for fix evaluation:**

```
In src/lib.rs, Spma::infer (around lines 158-232):

The cost table is built from old_patterns bit costs. Symbol IDs that
don't appear in old_patterns get cost 0.0, so unseen symbols contribute
nothing to E and never trigger is_anomaly.

Evaluate these fix options:

Option A — sentinel cost for unknown symbols:
  After building the cost table, assign a sentinel cost to any ID
  that still has cost 0.0 AND whose ID is >= the number of symbols
  seen during training (i.e. interned after load). The sentinel value
  could be the max bit cost seen in old_patterns, or a fixed value
  like the average cost. Simple, no API change.

Option B — reject unknown symbols before beam search:
  Before calling beam_search, check each id in `ids` against
  self.inner.original_alphabet. Any id not in the alphabet is
  immediately added to `unmatched` and flagged; set e_cost += some
  penalty per unknown symbol.

Option C — track max observed bit cost and use it as fallback:
  Store max_bit_cost on SpmaEngine during learn(). In infer(), any
  symbol with no cost entry gets max_bit_cost. Reflects "this symbol
  is at least as rare as the rarest known symbol."

Recommendation: Option B is the most principled — it uses
original_alphabet (already maintained) to distinguish known-but-rare
from truly-unseen, and keeps the cost table semantics clean.

Implementation sketch for Option B:
  1. In infer(), after building `ids`, iterate and check each against
     `self.inner.original_alphabet`.
  2. Collect unknown positions into a separate set.
  3. After beam_search, force covered[i]=false and add the symbol name
     to unmatched for each unknown position.
  4. Compute e_cost as sum of costs for uncovered positions PLUS a
     penalty for each unknown symbol (use average known bit cost as
     the penalty, or a configurable field on SpmaEngine).
  5. is_anomaly = e_cost > 0.0 (unchanged).

Add tests:
  - infer with one unseen symbol → is_anomaly=true
  - infer with all unseen symbols → is_anomaly=true, all in unmatched
  - infer with known-but-rare symbol → behavior unchanged
```

## 3. ~~MDL gate in `learn` rebuilds `current_multi` and `current_g` on every candidate iteration~~ **RESOLVED**

**Resolution (2026-07-16):** Hoisted `current_multi`/`current_g`/`current_e`/`current_t` before the
candidate loop. On acceptance, accumulators update in O(1) instead of rebuilding from `old_patterns`.
Dup check now tests against `current_multi` directly (avoids per-iteration id-vec allocation).
54 tests pass; added `mdl_accumulator_determinism_multi_candidate_corpus` to verify identical output.

---

## ~~3.~~ (archived) MDL gate in `learn` rebuilds `current_multi` and `current_g` on every candidate iteration

**Location:** `src/engine.rs`, `learn()` Pass 2, lines ~420-456

**Observed behavior:** The candidate evaluation loop rebuilds `current_multi` (multi-symbol grammar patterns as id vecs) and recomputes `current_g` from scratch on every iteration. `compute_total_e_dp` is also called twice per candidate (current and proposed). When a candidate is accepted and appended to `old_patterns`, the next iteration's rebuild correctly reflects it — but this means we pay O(grammar_size) per candidate just to maintain a view we could have kept as a running accumulator.

**Complexity:** O(C × G × L) per cycle, where C = candidates, G = grammar size (patterns), L = average pattern length. `compute_total_e_dp` is O(corpus_size × G × L) per call, called twice per candidate. At scale (large corpus, many candidates per cycle) this dominates runtime.

**Not a correctness issue.** Results are identical; this is purely a performance problem.

**Prompt for fix evaluation:**

```
In src/engine.rs, learn() Pass 2 (around lines 386-456):

The inner loop over sorted_candidates rebuilds current_multi and
current_g from self.old_patterns on every iteration. Since a candidate
is only appended to self.old_patterns when accepted, the rebuild is
needed to reflect prior acceptances within the same pass — but it still
pays O(grammar_size) scan unconditionally.

Refactor to maintain running accumulators:

1. Before the loop, compute once:
     let mut current_multi: Vec<Vec<u32>> = self.old_patterns
         .iter()
         .filter(|p| p.symbols.len() >= 2)
         .map(|p| p.symbols.iter().map(|s| s.name).collect())
         .collect();
     let mut current_g: f64 = current_multi.iter()
         .flat_map(|p| p.iter())
         .map(|&id| costs[id as usize])
         .sum();
     let mut current_e = compute_total_e_dp(&new_id_vecs, &current_multi, &costs);
     let mut current_t = current_g + current_e;

2. Inside the loop, compute new_multi/new_g/new_e/new_t as before
   (still O(G) for compute_total_e_dp — that's unavoidable for exact MDL).

3. When a candidate is accepted, update the accumulators instead of
   discarding them:
     current_multi.push(ngram.clone());
     current_g = new_g;
     current_e = new_e;
     current_t = new_t;
   Remove the self.old_patterns.push inside the loop and do it
   separately, or push and update in sync.

This reduces the per-candidate cost from O(G) rebuild + 2×compute_e_dp
to 1×compute_e_dp per candidate (the new_e check) plus O(1) accumulator
update on acceptance. The pre-loop compute_e_dp is paid once.

For a further speedup: the DP in compute_total_e_dp recomputes coverage
for all sentences from scratch when a single pattern is added. A truly
incremental approach would update only sentences that contain the new
ngram. That is a larger refactor — do it only if profiling shows
compute_total_e_dp is the bottleneck.

Verify correctness: the refactored pass must produce identical
final_patterns to the original for all existing test inputs.
Add a test that runs learn() on a corpus large enough to produce
multiple candidates per cycle and asserts the same grammar is produced.
```
