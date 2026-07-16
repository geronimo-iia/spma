# Non-Contiguous Patterns (Gap Matching) — Phase 2a

Specification for gap matching in beam search, grammar induction, and the miner
cold-start. Highest-ROI algorithmic addition: enables detecting "event A eventually
followed by event B" — the most common structural relationship in industrial fault logs.

Depends on:
- `docs/grammar-spec.md` — `Pattern::gaps: Vec<GapConstraint>`, `GapConstraint { min, max }`
- `docs/alignment-struct-design.md` — `Cell { is_gap, gap_span }` in `build_alignment`
- `docs/calibrated-score-design.md` — E_norm correct: gap-interior positions uncovered → contribute to E

No compatibility constraint with current code.

---

## What a gap pattern is

A pattern with a `Gap` symbol matches a New sequence where the symbols flanking the gap
appear within `max` positions of each other (with at least `min` positions in between).

```
Pattern: [TRIP_EVENT  Gap(min=0, max=5)  RESTORATION]
New:     [TRIP_EVENT  X  Y  RESTORATION  Z]
Match:   [TRIP_EVENT  .  .  RESTORATION  .]   ← gap consumed new[1..2], covered new[0] and new[3]
```

The gap symbol itself has no content — it is a placeholder that allows the beam to skip
New positions while staying "inside" the same pattern. Skipped New positions are
**uncovered** (contribute to E). Only the flanking OC symbols are covered.

This is different from simply having two separate patterns. A gap pattern asserts that
`TRIP_EVENT` and `RESTORATION` appear **in order** within a bounded window in the same
sequence. Two separate patterns only assert that both appear somewhere in the corpus.

---

## GapConstraint (replaces GapSpec)

Gaps are **not symbols** — they are constraints between adjacent symbols, stored as a
parallel vec `Pattern::gaps`. See `docs/grammar-spec.md` for the full `Pattern` definition.

```rust
pub struct GapConstraint {
    pub min: usize,  // minimum New positions to skip (0 = adjacent OK)
    pub max: usize,  // maximum New positions to skip (usize::MAX = unbounded)
}
```

`gaps` is empty for contiguous patterns. A pattern `[A Gap(0,3) B]` is stored as
`symbols=[A, B]`, `gaps=[GapConstraint{0,3}]`.

No consecutive-gap invariant — the parallel-vec structure makes it impossible.
No new beam state field for gap tracking — `can_extend` reads the constraint directly
from `pattern.gaps[old_pos - 1]` when advancing from symbol `old_pos-1` to `old_pos`.

---

## Beam search changes

### No new beam state

`PartialAlignment` gains **no new fields**. The gap constraint is read from
`pattern.gaps[old_pos - 1]` at the point `can_extend` is called.

### can_extend with gap constraint

When advancing to `old_pos` in pattern `oi`, check if there is a gap constraint between
`old_pos-1` and `old_pos`:

```rust
fn can_extend(&self, old_idx: usize, old_pos: usize, new_pos: usize, patterns: &[&Pattern]) -> bool {
    let pat = patterns[old_idx];
    match self.new_cursors.get(&old_idx) {
        None => old_pos == 0 && new_pos >= self.max_covered_new,
        Some(&prev_new) => {
            if pat.gaps.is_empty() || old_pos == 0 {
                // Contiguous: must be exactly next New position.
                new_pos == prev_new + 1
            } else {
                // Check gap constraint between symbols[old_pos-1] and symbols[old_pos].
                let gap = &pat.gaps[old_pos - 1];
                let skip = new_pos.saturating_sub(prev_new + 1);
                skip >= gap.min && skip <= gap.max
            }
        }
    }
}
```

No `extend_gap`, no `gap_can_consume`, no `gap_starts` HashMap. Gap positions are
**uncovered** — they are not recorded in `match_log` and contribute to E. `build_alignment`
infers gap cells from the distance between adjacent matched new positions.

---

## Grammar induction changes

### extract_learned_patterns — gap-aware version

Current: scans `covered` array for contiguous true spans, emits each span ≥ 2 as a
pattern.

New: scans for spans separated by short uncovered runs (up to `MAX_INDUCED_GAP`). If two
covered sub-spans are close enough, emit a single gap pattern instead of two separate
patterns.

```rust
pub const MAX_INDUCED_GAP: usize = 3;  // configurable per corpus

pub fn extract_learned_patterns(
    new_pattern: &Pattern,
    covered: &[bool],
    next_id: &mut u32,
    max_gap: usize,
) -> Vec<Pattern> {
    // Pass 1: find covered sub-spans (contiguous runs of true)
    let spans = contiguous_spans(covered);  // Vec<(start, end)>

    // Pass 2: merge adjacent spans separated by <= max_gap uncovered positions
    let merged = merge_close_spans(spans, covered.len(), max_gap);

    // Pass 3: for each merged span, emit a Pattern
    //   - Covered positions → OC symbols from new_pattern.symbols
    //   - Uncovered gaps within a merged span → Gap(min=0, max=actual_gap) symbol
    //   - Only emit if total OC symbol count >= 2
    for (sub_spans, gap_sizes) in merged:
        if total_oc_count < 2: skip
        let symbols = interleave(sub_spans, gap_sizes, new_pattern)
        result.push(Pattern::new(symbols, next_id))
}
```

**`merge_close_spans`** merges two spans `(a_start, a_end)` and `(b_start, b_end)` if
`b_start - a_end <= max_gap`. The gap size is `b_start - a_end`. Multiple spans merge
iteratively left-to-right.

**Example**:
```
covered: [T T T F F T T T]
spans:   [(0,3), (5,8)]
gap:     5 - 3 = 2 positions
max_gap: 3
→ merge → symbols=[sym0,sym1,sym2,sym5,sym6,sym7], gaps=[{},{},{GapConstraint{0,2}},{},{}]
  (gap constraint sits between index 2 and index 3 of symbols vec)
```

If gap > max_gap: emit two separate contiguous patterns (no gap). Gaps are empty vec.

### MAX_INDUCED_GAP configuration

`max_gap` is a parameter on `SpmaEngine`, settable before training:

```rust
impl SpmaEngine {
    pub fn set_max_induced_gap(&mut self, max: usize);
}
```

Default: 3. Reasonable for fault log corpora where events within a short window are
causally related. Set higher (5-10) for sparse logs with long inter-event intervals. Set
to 0 to disable gap induction entirely (contiguous patterns only).

---

## Miner cold-start changes

### extract_frequent_ngrams — gap-aware pairs

Current miner counts contiguous windows (bigrams, trigrams). Gap-aware miner additionally
counts co-occurring symbol pairs within a bounded window.

```rust
// For each sequence, for each position i, for each j in (i+1)..=(i+max_gap+1):
//   if symbols[i] and symbols[j] both appear >= min_freq times as a pair at this gap:
//     count (symbols[i], gap_size=j-i-1, symbols[j])
```

This produces candidates of the form `(sym_a, gap, sym_b)` with an associated frequency.
MDL selection applies the same gate: only add as a grammar pattern if it reduces global T.

Gap cost contribution to G: only the flanking OC symbols contribute to G (gap markers have
zero bit cost). T = G + E where E counts uncovered positions including gap-interior
positions. So a gap pattern is only MDL-viable if the flanking symbols co-occur frequently
enough that the savings on those positions outweighs the grammar cost of storing the
pattern.

### Candidate representation

Gap-aware candidates are `(Vec<u32>, Vec<GapConstraint>, u32)` — parallel vecs of symbol
IDs and gap constraints, plus frequency count. Converted directly to `Pattern { symbols,
gaps, frequency, ... }` before inserting into the grammar level.

---

## Alignment table output

Gap cells are synthetic — `build_alignment` inserts them when two adjacent match events
for the same pattern are separated by more than 1 New position. A gap cell has
`is_gap=true`, `gap_span = next_new_pos - prev_new_pos - 1`, `new_pos = prev_new_pos + 1`.

In `Display`, gap cells render as `<N>`:

```
         A    B    C    D    E
P1(L0)   A   <2>   .    D    .    ← gap of 2, A at new[0], D at new[3]
```

---

## Test strategy

### Unit: beam gap matching

```
// Pattern: symbols=[A,B], gaps=[{min:0,max:2}]
old = [Pattern{symbols:[A,B], gaps:[{0,2}]}], new = [A, X, B]
→ A at new[0], B at new[2]: skip = 2-0-1 = 1, in [0,2] → covered. E = cost(X).

old = [Pattern{symbols:[A,B], gaps:[{0,2}]}], new = [A, X, Y, Z, B]
→ B at new[4]: skip = 4-0-1 = 3, > max=2 → NOT covered.
→ best: A alone or B alone. E > 0.

old = [[A, Gap(0,2), B]], new = [B, A]
→ inter-pattern ordering: A at new[1] would start gap at new[2] but B already at new[0].
→ NOT covered in order. E = cost(A) + cost(B).
```

### Unit: extract_learned_patterns with gaps

```
covered = [T T F F T T], max_gap = 3
→ gap = 2, <= max_gap → symbols=[sym0,sym1,sym4,sym5], gaps=[{},{GapConstraint{0,2}},{}]

covered = [T T F F F T T], max_gap = 3
→ gap = 3, <= max_gap → symbols=[sym0,sym1,sym5,sym6], gaps=[{},{GapConstraint{0,3}},{}]

covered = [T T F F F F T T], max_gap = 3
→ gap = 4, > max_gap → two contiguous patterns: symbols=[sym0,sym1] gaps=[], symbols=[sym6,sym7] gaps=[]
```

### Integration: gap pattern learned from corpus

```rust
// Corpus: 10× ["TRIP", "X", "RESTORATION"] (X varies each time)
// max_induced_gap = 1
// Expected: grammar learns Pattern{symbols:[TRIP,RESTORATION], gaps:[{0,1}]}
// Infer ["TRIP", "Y", "RESTORATION"] → skip=1 in [0,1] → covered. E = cost(Y).
// Infer ["RESTORATION", "TRIP"] → is_anomaly (wrong order)
// Infer ["TRIP", "A", "B", "RESTORATION"] → skip=2 > max=1 → not covered
```

---

## What this does NOT change

- MDL objective (T = G + E) — unchanged. Gaps have zero grammar cost; only OC symbols
  count toward G.
- Hierarchical levels — gap patterns at level 0 can be referenced by level-1 patterns
  exactly like contiguous patterns (they have a pattern ID like any other pattern).
- Serialization — `Pattern::gaps: Vec<GapConstraint>` serializes as a field on `Pattern`. Deferred like all serialization.
- `E_norm` computation — gap-interior positions are uncovered → contribute to E → correct.
