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
