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
- `extract_learned_patterns` now returns all maximal contiguous covered spans as separate
  patterns; previously non-contiguous covered positions were collapsed into one pattern,
  producing invalid sequences for DP tiling
- `Spma::infer` now correctly fires `is_anomaly` for symbols never seen during training;
  unknown symbols are forced uncovered and accumulate an `unknown_penalty` (average known
  bit cost) rather than silently receiving cost 0.0

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
