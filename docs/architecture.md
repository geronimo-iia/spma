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

**`PartialAlignment` state:**
- `old_cursors: HashMap<usize, usize>` — last matched Old position per Old pattern (monotonicity)
- `new_cursors: HashMap<usize, usize>` — last matched New position per Old pattern (span contiguity)
- `max_covered_new: usize` — highest New position covered by any Old pattern (inter-pattern ordering)
- `covered_new: Vec<bool>` — which New positions are covered

**Monotonicity constraint**: each Old pattern can only advance forward within its own symbol sequence. No symbol in an Old pattern is matched at a position earlier than its previous match in that pattern.

**Span contiguity / gap constraint** (`new_cursors`): when advancing to the next symbol of an Old pattern, the default is exactly `prev_new + 1`. If the pattern has a `GapConstraint` at that position (`Pattern::gaps[old_pos-1]`), the beam allows `skip` New positions in `[min, max]` instead. Skipped positions are uncovered (contribute to E). The first symbol of a pattern can start at any New position.

**Inter-pattern ordering constraint** (`max_covered_new`): the first symbol of a new Old pattern must begin at a New position `>= max_covered_new`. Prevents mid-stream interleaving; does not detect full-sequence reorderings — see [docs/known-limitations.md](known-limitations.md).

**Match log / MatchArena**: beam search owns a `MatchArena` (flat `Vec<MatchNode>` linked list). Each `PartialAlignment` holds a single `u32` tail index rather than a cloned `Vec<MatchEvent>`. Forking copies one `u32`. After beam completion, `arena.collect(winning.log_tail)` walks the linked list once to reconstruct all match events for the winning alignment.

**Why not pairwise**: the original implementation matched New against one Old pattern at a time and merged results. This is wrong — SPMA's compression gain comes from using multiple Old patterns to cover different spans of New simultaneously. Pairwise alignment misses cross-pattern coverage and systematically underestimates CD.

## Corpus costs fallback

Symbols present in the training corpus but never absorbed into any grammar pattern would have `bit_cost = 0.0` in the Old store, making them uncovered-but-free at inference (E=0 false negatives). Fix: `SpmaEngine::corpus_costs: Vec<f64>` snapshots Shannon costs from `new_patterns` at the end of `learn()`. At inference, `costs[id]` falls back to `corpus_costs[id]` when the grammar has no cost for that symbol. Serialized in `GrammarSnapshot`; `load()` restores `original_alphabet` from all interned names so corpus-known symbols are not misclassified as unknown. (Resolves Issue #4.)

## Learning loop

In `src/engine.rs`:

1. Cold start: n-gram miner bootstraps the Old store from frequent bigrams/trigrams when grammar is empty (no Old patterns yet).
2. Once grammar is non-empty: beam search drives extraction. For each New pattern, run beam search; if best alignment has CD > 0, extract covered subsequence as a new Old pattern if it passes MDL gate.
3. Convergence: loop terminates when `old_grew || added_this_cycle` is false — grammar stopped growing and no new patterns were added this pass. No T-epsilon check (unreliable because cost recomputation between measurements produces noise that never converges).

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

Grammar persistence: serde + bincode serialisation of the Old pattern store. `GrammarSnapshot` includes `old_patterns`, `interner_names`, and `corpus_costs`. On load, `original_alphabet` is populated from all interned names.

## Hierarchical grammar (N-level)

Implemented. The engine runs N levels: level-0 induces atom patterns; level-k induces patterns over level-(k-1) pattern IDs. `SymbolRef::Pattern(pid)` references a lower-level pattern. `InferResult::level_e_norms` gives per-level normalized E. See [docs/grammar-spec.md](grammar-spec.md) for the data model.

E_norm is the primary anomaly signal: `E / raw_new_cost`, range [0.0, 1.0]. See [docs/scoring.md](scoring.md).

## Test organisation

Integration tests split into four modules under `tests/`:

| File | Scope |
|---|---|
| `tests/symbols.rs` | `Symbol`, `Pattern`, `Interner`, `compute_t_ge`, Shannon bit costs |
| `tests/engine.rs` | `SpmaEngine` internals: learning cycle, MDL gate, compression ratio, convergence |
| `tests/beam.rs` | `beam_search`, `write_alignment_table`, span contiguity, inter-pattern ordering |
| `tests/api.rs` | Public `Spma` API: train/infer/save/load, corpus_costs fallback, grammar_size |

Unit tests for `beam_search` (including contiguity edge cases) live in `src/beam.rs`.

