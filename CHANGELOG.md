# Changelog

## [0.2.0] — 2026-07-18

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
