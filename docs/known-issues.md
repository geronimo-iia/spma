# Known Issues

Three confirmed deviations from Wolff's SP theory, discovered via theory-vs-implementation audit
(2026-07-16). Issues are ordered by fix dependency: #1 must be fixed before #2 can be validated,
and both before #3 is meaningful.

## 1. Single-symbol patterns in grammar — theory violation that blocks multi-symbol learning

**Status:** Open — architectural. Root cause of issues #2 and #3 being unobservable.

**Theory says:** Individual symbols are atomic alphabet elements, implicit throughout the system.
An SP-pattern is by definition an array of multiple symbols. The Old store holds grammar patterns,
not alphabet atoms. (Wolff: *"each SP-pattern is an array of atomic SP-symbols"*.)

**Implementation does:** `learn()` at `src/engine.rs:318–329` explicitly seeds `old_patterns`
with one length-1 pattern per unique symbol before the learning loop starts. These singleton
patterns make E=0 for any known-symbol input immediately, so the MDL gate permanently rejects
every multi-symbol candidate (adding `[A,B]` raises G by `cost(A)+cost(B)` but cannot reduce
E which is already 0). Grammar is stuck at singletons forever.

**Downstream effects:**
- Order violations undetectable (E=0 for any permutation of known symbols)
- `beam_search` span-matching issue (#3) is unobservable — no multi-symbol patterns exist to test against
- `max_cycles` premature termination (#2) may mask itself — loop exits "converged" when nothing can change

**Prompt for fix:**

```
In src/engine.rs, learn():

The block at lines 318–329 seeds old_patterns with one length-1 pattern per unique symbol.
This violates SP theory (individual symbols are alphabet atoms, not grammar entries) and
causes the MDL gate to permanently reject all multi-symbol candidates.

Remove the singleton seeding block entirely (lines 318–329 and the apply_symbol_costs call
at line 331 that follows it).

The alphabet is already tracked in self.inner.original_alphabet (lib.rs) for unknown-symbol
detection — no information is lost.

After removal:
  1. The n-gram cold-start miner (extract_frequent_ngrams) must bootstrap the grammar from
     scratch. It already does this — verify it runs and inserts multi-symbol patterns.
  2. Bit costs: collect_frequencies builds costs from old_patterns + new_patterns. With no
     singletons in old_patterns, initial costs come from new_patterns alone — this is correct
     (corpus frequencies, not grammar frequencies).
  3. E will be > 0 for all inputs until multi-symbol patterns form. This is correct behavior.
  4. Unknown-symbol detection (original_alphabet check in lib.rs) is unaffected.

Tests to add:
  - Train on ["A","B","C"] × 5; assert at least one old_pattern has len() >= 2.
  - Train on ["A","B","C"] × 5; infer ["C","B","A"]; assert is_anomaly=true.
  - Existing tests may break if they assumed E=0 for known sequences — audit and fix
    expected values to reflect the corrected behavior.

Verify examples/fault_detection.rs still produces sensible output (anomaly for unknown
fault types, OK for known sequences once grammar stabilises).
```

## 2. `max_cycles` hard cap truncates learning prematurely

**Status:** Open — moderate. Observable only after issue #1 is fixed (grammar can grow).

**Theory says:** Iterate until T stops decreasing. No cycle limit exists in Wolff's formulation.
Convergence is defined purely by the MDL objective.

**Implementation does:** `learn()` at `src/engine.rs:493` breaks the loop when
`total_cycles >= self.max_cycles` (default 10). On a large or varied corpus, 10 cycles may
not be enough for the grammar to converge — learning halts while T is still decreasing.

**Prompt for fix:**

```
In src/engine.rs, learn() and SpmaEngine::new():

The loop at lines 486–495 breaks on total_cycles >= self.max_cycles (default 10).
This is not in Wolff's theory. The correct termination condition is T stops decreasing.

Options:
  A) Remove max_cycles entirely. Use only the T-convergence and no-improvement checks
     already present (lines 487–491). Risk: runaway on adversarial input.
  B) Keep max_cycles as a safety valve but raise the default to 1000 and document
     that it should never be the binding constraint in normal use.

Recommendation: Option B. A cycle cap is a reasonable safeguard; the problem is the
default of 10, not the mechanism.

Steps:
  1. Change default max_cycles from 10 to 1000 in SpmaEngine::new().
  2. Add a log/eprintln warning if termination was due to max_cycles (not convergence),
     so users know learning was truncated.
  3. Add a test: construct a corpus that requires > 10 cycles to converge; assert the
     grammar contains patterns that would only appear after cycle 11+.

Note: fix issue #1 first. With singletons blocking multi-symbol patterns, the loop
converges in 1–2 cycles regardless of max_cycles, so this issue is unobservable until #1
is resolved.
```

## 3. Beam search matches symbols scatter-style — span contiguity not enforced

**Status:** Open — architectural. Unobservable until issue #1 is fixed.

**Theory says:** SP-multiple-alignment aligns contiguous spans of New against contiguous spans
of Old. A multi-symbol Old pattern `[A, B, C]` should only match a contiguous block `A B C`
in New — the symbols must appear adjacent and in order. The alignment is a set of column
bindings between New positions and Old positions; each Old pattern occupies a contiguous
column range.

**Implementation does:** `beam_search` in `src/beam.rs` matches one symbol of New per beam
step. The monotonicity constraint (`can_extend`, line 57) only prevents an Old pattern's
cursor from going backwards within that pattern — it does not require the matched New
positions to be contiguous. A pattern `[A, B]` can match `A` at New position 0 and `B` at
New position 5, with unrelated symbols at positions 1–4. This is scatter-matching, not
span-matching.

**Consequence:** once issue #1 is fixed and multi-symbol patterns form, the beam will
incorrectly allow non-contiguous matches, inflating coverage (E artificially low) and
producing alignment tables that don't correspond to valid SP-multiple-alignments.

**Prompt for fix:**

```
In src/beam.rs, PartialAlignment and beam_search():

The monotonicity constraint (can_extend, line 57) prevents an Old pattern from matching
backwards but does not require contiguous New positions. A pattern [A, B] can match
A at New[0] and B at New[5], which violates SP theory's span-contiguity requirement.

Fix: when extending a match for old pattern `oi` at new position `new_pos`, require
that new_pos == last_new_pos_for_oi + 1 (or new_pos == 0 for the first match).

Data structure change needed:
  - PartialAlignment currently tracks old_cursors: HashMap<usize, usize> (old_idx → old_pos).
  - Add new_cursors: HashMap<usize, usize> (old_idx → last_matched_new_pos).
  - In extend_match: before accepting, check
      new_cursors.get(old_idx).map_or(true, |&prev_new| new_pos == prev_new + 1)
  - Update new_cursors on accept.

Edge case: the FIRST symbol of an Old pattern matched against New can start at any New
position (no contiguity requirement for the start). Contiguity is only required for
subsequent symbols of the same Old pattern.

After fix:
  - A pattern [A, B, C] can only match "A B C" as a block, not "A ... B ... C".
  - Alignment tables will be more sparse (higher E) but more semantically correct.
  - Add a test: old = [[A, B]], new = [A, X, B]; assert covered = [true, false, false]
    (B at position 2 is NOT covered because it's not contiguous with A at position 0).
  - Add a test: old = [[A, B]], new = [A, B, C]; assert covered = [true, true, false].

Note: this is the largest of the three fixes — it changes the core beam scoring logic.
Fix issues #1 and #2 first and validate that multi-symbol grammars form correctly before
tackling this one.
```
