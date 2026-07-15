# Changelog

## [Unreleased]

### Added
- String interning: `Symbol::name` → `u32` ID, `Interner` struct
- Correct Shannon bit costs: `cost(s) = -log2(freq/total)`, removed `cost_factor` distortion
- T=G+E decomposition: G charged once at insertion
- Beam search with monotonicity constraint
- `--no-learn` inference mode + grammar persistence (serde + bincode)
- Learning loop with MDL gate, alignment table printer
- public API now `Spma` / `InferResult`

### Fixed
- Learning loop convergence: replaced broken T-epsilon check (unreliable because cost
  recomputation between `t_before`/`t_after` produced noise deltas that never hit 1e-6)
  with grammar-stability check (`old_grew || added_this_cycle`); loop now terminates in
  O(grammar-growth) cycles instead of always hitting `max_cycles`; 100× faster test suite
- Zero-cost symbol false negatives: `corpus_costs` field added to `SpmaEngine`; snapshotted
  from `new_patterns` at end of `learn()`; used as fallback in `infer()` for symbols not
  absorbed into any grammar pattern; serialized in `GrammarSnapshot`; `load()` now populates
  `original_alphabet` from all interned names, not just `old_patterns`
- Inter-pattern ordering partially enforced: `max_covered_new: usize` added to `PartialAlignment`;
  first symbol of any new Old pattern requires `new_pos >= max_covered_new`; prevents a second
  pattern from starting behind the current New frontier; full sequence reorderings remain
  undetected (see docs/known-issues.md #5)
- Beam search span contiguity enforced: `new_cursors` added to `PartialAlignment`; advancing
  within a multi-symbol Old pattern now requires `new_pos == prev_new + 1`; scatter-matching
  `[A,B]` against `[A,X,B]` no longer covers `B`
- `max_cycles` default raised 10 → 1000; truncation warning emitted when cap is hit; `set_max_cycles`
  / `grammar_size` added to public `Spma` API
- `extract_learned_patterns` now returns all maximal contiguous covered spans as separate
  patterns; previously non-contiguous covered positions were collapsed into one pattern,
  producing invalid sequences for DP tiling
- `Spma::infer` now correctly fires `is_anomaly` for symbols never seen during training;
  unknown symbols are forced uncovered and accumulate an `unknown_penalty` (average known
  bit cost) rather than silently receiving cost 0.0
- Removed singleton seeding from `learn()` — individual symbols are alphabet atoms per SP
  theory, not grammar entries; seeding them into `old_patterns` made the MDL gate reject
  all multi-symbol candidates (E=0 trivially with singletons, adding any pattern raised T)
- Convergence check now uses `compute_total_e_dp`-based T (consistent with MDL gate) instead
  of `beam_search`-based T; the two disagreed, causing spurious convergence and T non-monotone
  warnings after singleton removal

### Changed
- Integration tests split from monolithic `tests/integration_tests.rs` into four focused
  modules: `tests/symbols.rs`, `tests/engine.rs`, `tests/beam.rs`, `tests/api.rs`; 69 tests
  across 4 suites

### Performance
- MDL Pass 2 in `learn()` hoists `current_multi`/`current_g`/`current_e`/`current_t`
  before the candidate loop and updates them incrementally on acceptance; previously rebuilt
  from `old_patterns` on every iteration (O(C×G) → O(G) + O(C) amortised)

### Removed
- Dead HitNode-based scaffolding: `run_recognition_cycle`, `find_hits`, `build_hit_structure`,
  and associated `SpmaEngine` fields (`patterns`, `alignments`, `parsing_alignments`, etc.)
- Dead model types: `Alignment`, `AlignmentElement`, `HitNode`, `Grammar` and all associated
  methods — none were reachable from the live beam-search path
- `LearningResults.alignments` and `LearningResults.grammars` fields (always `vec![]`)
