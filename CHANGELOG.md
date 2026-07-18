# Changelog

## [Unreleased]

### Added

- `Spma::retrain` — extends an already-trained model with a new batch of sequences
  without discarding prior grammar; cumulative atom frequencies persisted on `Spma`
  and serialized with the model
- CLI `retrain` subcommand: `spma retrain --model model.json --corpus new_normal.txt`

### Changed

- `Spma::infer`: reuses level-0 `match_log` for level-1 pid_seq extraction; eliminates
  one redundant `beam_search` call per inference sequence
- `GrammarLevel`: `SymbolIndex` inverted index built once per level, eliminates per-call
  `HashMap` rebuild in `beam_search`; not serialized, rebuilt on `load`
- `Spma::load`: rebuilds `SymbolIndex` on each grammar level after deserialization
- `Spma::recalibrate`: rebuilds `SymbolIndex` on each level before running inference
- `atom_freq` and `total_symbol_count` fields now `#[serde(default)]` — existing saved
  models load without error

### Changed

- Cargo workspace split: `spma` lib crate (no `anyhow`/`clap`/`mimalloc`) + `spma-cli` bin crate; install with `cargo install spma-cli`
- `anyhow`, `clap`, `mimalloc` removed from library dependency graph — no longer forced on downstream lib users

## [0.1.0] — 2026-07-17

### Added

- Hierarchical grammar induction: N-level MDL-gated beam search produces grammar levels where each level's patterns reference the one below
- MDL scoring (T = G + E): frequency-based Shannon bit costs; G charged once per pattern insertion
- Per-level anomaly gate: `set_level_threshold(level, t)` gates `is_anomaly` independently at each grammar level
- Gap matching: patterns with induced gap constraints (e.g. `TRIP ~[0,3]→ RESTORE`) learned from co-occurring symbols with variable-length fillers
- Calibrated E_norm distribution: `EDistribution` fit from training corpus; `anomaly_percentile` and configurable threshold
- `Spma::recalibrate` subcommand and method: refit `e_distribution` on new normal data without re-training the grammar
- Parallel inference via rayon: `infer` on batched input uses all available cores
- `Alignment` display: per-symbol coverage table with E/CD/T footer
- `Spma::save` / `Spma::load`: JSON serialization via serde
- CLI subcommands: `train`, `infer`, `recalibrate`, `grammar` (human and JSON output)
- Validated on LogHub HDFS: F1 = 0.893 unsupervised (see spma-experiments)

### Fixed

- `Alignment` display: T footer now correctly shows E + CD (was showing E twice)
