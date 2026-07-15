# Architecture


## String interning

All symbol names are interned to `u32` IDs at load time. Symbol comparison is a single integer equality — no string allocation, no memcmp, SIMD-ready.

The `Interner` struct (`src/intern.rs`) maps `&str → u32` and back. All hot paths operate on `u32` IDs. Display functions receive `&Interner` to resolve names for output.

**Why**: the original implementation used `String` fields on every `Symbol`. Hit detection was O(n × m × string_length). With interning, comparison is O(1).

## Bit costs — Shannon entropy

Every symbol has a bit cost: `cost(s) = -log2(freq(s) / total_freq)`.

High-frequency symbols are cheap. Rare symbols are expensive. Costs are computed from the full corpus before learning starts and recomputed after each grammar update.

**Why not `cost_factor` multiplier**: the original code multiplied DataSymbol costs by a tunable `cost_factor`. This breaks the information-theoretic meaning of T=G+E. The Shannon formula is the correct objective; empirical tuning on top of it is distortion, not calibration.

## T=G+E scoring

The scoring objective from MDL / Wolff:

- **G** = grammar cost — charged **once at insertion** when a pattern is added to the Old store. Not re-charged on each alignment that uses it.
- **E** = encoding cost — sum of bit costs of New symbols **not covered** by any matched Old pattern.
- **T** = G + E
- **CD** (compression difference) = raw cost of New − T. Positive CD means the alignment improves compression.

**Critical design decision**: G=0 per alignment for patterns already in the grammar. When a pattern is first added to Old store, its cost is paid. Subsequent uses are free. This is what makes CD positive for genuine matches — if G were re-charged per alignment, CD would approach zero always (matched symbol costs ≈ grammar symbol costs), and the learning signal would vanish.

The displayed `G (grammar patterns used)` in alignment tables is a display-only sum of bit costs of matched Old patterns — it does not affect scoring.

## Beam search (SPMA core)

Implemented in `src/beam.rs`. One New pattern aligned against all Old patterns simultaneously, building left-to-right.

**Algorithm**: staged beam search.
1. Start with one empty `PartialAlignment`
2. For each position `p` in New:
   - Option A: skip (no match at `p`, symbol goes to E)
   - Option B: match `new[p]` against each Old pattern at each position where `old[q] == new[p]`
3. Prune to top-K by CD after each position
4. Return top-K complete alignments sorted by CD

**Monotonicity constraint**: each Old pattern can only advance forward. No symbol in an Old pattern is matched at a position earlier than its previous match in that pattern. This preserves left-to-right order in every row of the alignment table.

**Why not pairwise**: the original implementation matched New against one Old pattern at a time and merged results. This is wrong — SPMA's compression gain comes from using multiple Old patterns to cover different spans of New simultaneously. Pairwise alignment misses cross-pattern coverage and systematically underestimates CD.

## Learning loop

In `src/engine.rs`:

1. Cold start: n-gram miner bootstraps the Old store from frequent bigrams/trigrams when grammar is empty.
2. Once grammar is non-empty: beam search drives extraction. For each New pattern, run beam search; if best alignment has CD > 0, extract covered subsequence as a new Old pattern (if it passes MDL gate).
3. Convergence: loop until T stops decreasing across a full pass over New patterns.

**One-trial learning**: a New pattern presented once is immediately added to Old store as a BASIC_PATTERN. Second presentation finds it in the grammar, CD > 0, full match.

**Add-only store**: Old patterns are never deleted. This gives structural immunity to catastrophic forgetting — the property EWC (2017) buys at high cost, SPMA gets for free.

## Alignment table printer

The alignment table IS the explanation. No post-hoc attribution needed.

```
New:   the  cat  sat  on   the  mat
Old1:  the            -    the           [the]
Old2:       cat                          [cat]
Old3:            sat                     [sat]
Old4:                 on                 [on]
Old5:                          mat       [mat]

Matched: 6/6  G=14.2 bits (grammar patterns used)  E=0.0 bits  CD=+31.1 bits
```

Each column = one position in New. Each row = one Old pattern used. Unmatched New symbols show as `[UNMATCHED:symbol]` with no Old pattern beneath. `E > 0` and `CD < 0` are the anomaly signal for UC1.

## Train/test workflow

```bash
spma train normal_logs.txt                          # trains grammar, saves ./spma_grammar.bin
spma infer anomaly.txt                              # loads grammar, aligns, no update
spma infer --grammar /path/custom.bin anomaly.txt
```

`--no-learn` is essential for UC1 (anomaly detection). Without it, every new symbol is added to the grammar on first sight, making E=0 always and destroying the anomaly signal.

Grammar persistence: serde + bincode serialisation of the Old pattern store.

