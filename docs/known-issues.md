# Known Issues

## 1. Order violations undetectable — single-symbol grammar monopoly

**Status:** Open — architectural

### Observed behavior

Training on a corpus of identical ordered sequences (e.g. `A B C D` repeated N times)
produces a grammar of only single-symbol patterns `{A, B, C, D}`. A reordered input
`D C B A` scores E=0, `is_anomaly=false` — indistinguishable from the training sequence.

### Root cause

Two compounding problems:

**Problem A — MDL gate blocks multi-symbol patterns once singletons exist.**
Cold-start n-gram miner proposes bigrams like `[A, B]`. MDL check compares:

```
current_t = G_singletons + E
proposed_t = G_singletons + cost(A) + cost(B) + E'
```

Once every symbol has a single-symbol grammar entry, E=0 for any known sequence.
Adding `[A, B]` to G raises G by `cost(A)+cost(B)` but cannot reduce E (already 0).
`proposed_t > current_t` → rejected. This holds for every multi-symbol candidate.
Grammar is permanently stuck at single-symbol patterns.

**Problem B — beam search is order-agnostic.**
Even if multi-symbol patterns did exist, `beam_search` matches symbol IDs against
Old patterns; the monotonicity constraint only prevents an Old pattern from matching
*backwards within itself* — it does not constrain which positions of New a pattern
must align to. Single-symbol patterns have no internal order to enforce, so they
match any permutation freely.

### What doesn't help

- Boundary markers (`<` / `>`) — they become single-symbol grammar entries too;
  same MDL trap.
- More training data — repetition reinforces singletons, doesn't break the deadlock.
- The `unknown_penalty` fix — only fires for symbols absent from `original_alphabet`;
  reordering uses known symbols.

### Fix directions

**Direction A — suppress single-symbol grammar entries.**
Never add length-1 patterns to `old_patterns`. Forces the MDL gate to evaluate
multi-symbol candidates against a grammar that cannot yet cover every symbol for
free. Risk: E is always > 0 for any input until multi-symbol patterns develop;
threshold calibration needed.

**Direction B — positional encoding in beam search.**
Add expected-position metadata to grammar patterns. Penalise alignments where
`new_position` deviates significantly from `expected_position`. Requires changing
`BeamAlignment` scoring and breaking the current position-agnostic abstraction.

**Direction C — sequence-level MDL using ordered patterns only.**
Replace the n-gram miner with an ordered-subsequence miner that only proposes
patterns where the symbols appear in the same relative order as in training. Multi-
symbol patterns then carry implicit order. Still blocked by Problem A until
single-symbol suppression (Direction A) is also applied.

**Recommended starting point:** Direction A alone is a one-line change to the
cold-start path and the beam-extraction acceptance in `learn()`. Evaluate whether
it produces stable multi-symbol grammars before tackling beam scoring.

### Prompt for fix evaluation

```
In src/engine.rs, learn():

Single-symbol grammar patterns make the MDL gate permanently reject multi-symbol
candidates on any corpus where all symbols appear in isolation at least once.

Evaluate Direction A: prevent length-1 patterns from being added to old_patterns.

Affected sites:
  1. Cold-start n-gram miner (Pass 1): already filters len() >= 2 on extracted spans
     via extract_learned_patterns — check whether any path still inserts singletons.
  2. One-trial learning: new patterns (BASIC_PATTERNs) from the input corpus are
     added directly to old_patterns in learn(); single-symbol input sequences would
     add singletons here.
  3. MDL acceptance (Pass 2): ngrams are already >= 2 symbols (from extract_learned_patterns)
     — no change needed here.

Steps:
  1. Audit every self.old_patterns.push(...) call and add a guard:
       if pat.symbols.len() < 2 { continue; }  // or skip
  2. Run the existing 54 tests — some may break if they relied on single-symbol
     grammar entries for coverage.
  3. Run examples/fault_detection.rs — check whether multi-symbol patterns now form
     and whether E > 0 for unknown-symbol inputs still fires correctly.
  4. Add a test: train on ["A","B","C"] repeated 5 times; assert old_patterns
     contains at least one pattern with len() >= 2.
  5. Add a test: train on ["A","B","C"] repeated 5 times; infer ["C","B","A"];
     assert is_anomaly=true (order violation detected via multi-symbol pattern coverage).

Consider: if all input sequences are length-1 (degenerate corpus), the grammar
would be empty. Decide whether that is acceptable or whether a fallback is needed.
```
