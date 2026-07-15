# Known Issues

Discovered via theory-vs-implementation audit (2026-07-16) and example-driven testing.

## 5. Multi-pattern stitching defeats global order detection

**Status:** Partially mitigated (2026-07-16) — intra-pattern interleaving blocked; inter-pattern
reordering open. Two fix strategies documented below; neither yet implemented.

### Context

SP theory (Wolff) operates at multiple levels of abstraction simultaneously. The current
implementation is a **level-1** system: atomic symbols form patterns, patterns cover New
sequences. This is sufficient for detecting unknown symbols and missing structure. It is
insufficient for detecting **reorderings** of known structure.

The beam operates symbol-by-symbol, left-to-right on New. It has no concept of "pattern X
should precede pattern Y." Each Old pattern is matched independently. Because beam processing
is inherently left-to-right, a New sequence `[C, D, A, B]` covered by patterns `[C,D]` and
`[A,B]` looks identical to `[A, B, C, D]` from the beam's perspective — both patterns start
at a position ≥ the current frontier, so `max_covered_new` (partial fix) is trivially
satisfied in both cases.

**Partial fix applied:** `max_covered_new` in `PartialAlignment` blocks a pattern from
starting at a New position that was already passed. This catches mid-stream interleaving
(pattern 2 starting before pattern 1 finishes) but cannot detect whole-sequence reorderings
because the patterns always encounter the frontier in left-to-right order regardless of
training order.

**Root cause:** The ordering constraint is a property of the *grammar* — specifically, the
relationship between patterns. The beam only knows individual patterns. Fixing this requires
lifting ordering knowledge out of the beam and into the grammar representation.

### Fix A — Post-beam ordering penalty (pragmatic, limited to 2 levels)

After beam alignment, extract the sequence of Old patterns used, ordered by their starting
New position. Compare against the dominant ordering observed during training (recorded as
pairwise counts: "in training, pattern i started before pattern j in N of M sequences").
Pairs whose inference order contradicts the dominant training order contribute a penalty to
a separate `e_order: f64` field in `InferResult`.

**Implementation:**

```
Training side (engine.rs, after convergence):
  pattern_order: HashMap<(u32, u32), (u32, u32)>
  //              (pat_i_id, pat_j_id) → (count_i_before_j, count_j_before_i)
  For each training sequence, run beam, extract pattern start positions,
  record pairwise orderings.

Inference side (lib.rs, infer()):
  After beam, sort used patterns by starting New position.
  For each adjacent pair (or all pairs), look up pattern_order.
  If inference order contradicts dominant training order by > threshold:
    e_order += log_odds_penalty(count_expected, count_observed)

InferResult gains:
  e_order: f64                          // ordering anomaly cost
  ordering_violations: Vec<(String, String)>  // pattern name pairs
```

**Pros:** No beam changes. Clean separation. Calibrated via log-odds.

**Cons:**
- Strictly 2-level. Pattern sequences are compared but the patterns themselves have no
  ordering relative to their internal symbols. A 3-level reordering (symbols within
  sub-sequences, sub-sequences within sequences, sequences within episodes) requires
  3 separate penalty layers, each added manually.
- `e_cost` and `e_order` are different quantities with different units — combining them
  into a single anomaly score requires an arbitrary weighting.
- Pairwise pattern ordering is fragile when the grammar contains many short patterns:
  a 4-symbol sequence covered by 4 bigrams produces 6 pairs with uncertain ordering
  statistics, especially on small corpora.

### Fix B — Grammar-level sequence patterns (theoretically correct, full N levels)

Extend the grammar to hold patterns-of-patterns. After level-1 learning converges, run a
**level-2 learning pass**: treat each training sequence as a sequence of pattern IDs (the
patterns that covered it at level 1), and run the same n-gram miner + MDL gate + beam on
those ID sequences. The result is level-2 patterns — ordered sequences of level-1 pattern
references. Recurse to level 3 if needed.

This is what SP theory specifies. Wolff: *"SP-multiple-alignment works at all levels of
a processing hierarchy using the same mechanism."* Ordering is not a special case — it is
a coverage failure at the next level up, caught by the same beam.

**Structural change to `Symbol`:**

```rust
// Current
pub struct Symbol { pub name: u32, ... }  // name = atomic symbol ID

// Required
pub enum SymbolRef {
    Atom(u32),     // atomic symbol
    Pattern(u32),  // reference to an Old pattern ID
}
pub struct Symbol { pub name: SymbolRef, ... }
```

**`learn()` gains a second phase:**

```
Phase 1 (current): atomic symbols → old_patterns (level-1 grammar)
Phase 2 (new):
  For each training sequence:
    Run beam with level-1 grammar → get pattern coverage sequence [P3, P7, P2, ...]
  Treat [P3, P7, P2, ...] as a new "pattern-ID sequence"
  Run same n-gram miner + MDL gate on pattern-ID sequences
  Result: level-2 patterns, e.g. [[P3, P7], [P2, P5]] → level-2 old_patterns
Phase 3+ (optional): recurse on level-2 pattern-ID sequences
```

**Beam becomes recursive:** When beam encounters a `SymbolRef::Pattern(id)`, it
recursively aligns the corresponding New sub-sequence against the referenced pattern.
Coverage at level N implies coverage at level N+1.

**Pros:**
- Full N-level ordering. Arbitrarily deep hierarchy.
- No separate penalty mechanism — ordering violation IS an E cost, in the same units,
  with the same semantics as symbol-level E.
- Self-similar: same algorithm, same MDL gate, same beam at every level.
- Grammar grows to represent actual structure (sub-sequences, episodes, etc.).

**Cons:**
- Major redesign. `Symbol`, `beam_search`, `learn()`, `GrammarSnapshot`, `infer()` all change.
- Level-2 beam needs level-1 beam as a subroutine — inference cost multiplies.
- Small corpora may not produce stable level-2 patterns (MDL gate rejects if pattern-ID
  sequences are too varied).
- `grammar_size()`, alignment table printing, and all existing tests need updating.

### Recommendation

Fix A for an immediate pragmatic gain with minimal code change. Fix B for a correct
implementation that aligns with SP theory and handles arbitrary depth. They are not
mutually exclusive — Fix A can be removed once Fix B is implemented, since Fix B subsumes it.


### Tests to add (Fix A)

```
- Train on [[A,B,C,D]]×10. Grammar learns [A,B] and [C,D].
  Infer [C,D,A,B] → e_order > 0, e_cost == 0.
  Infer [A,B,C,D] → e_order == 0, e_cost == 0.

- Train on [[A,B],[C,D]]×10 and [[C,D],[A,B]]×10 (mixed order corpus).
  Infer [C,D,A,B] → e_order ≈ 0 (no dominant order).
```

### Tests to add (Fix B)

```
- After level-2 learning, grammar_level2_size() > 0.
- Infer [C,D,A,B] when trained on [A,B,C,D]×10 → is_anomaly=true, E > 0.
- Infer [A,B,C,D] → is_anomaly=false, E == 0.
```
