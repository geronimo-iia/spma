# Profiling Guide

## Setup (one-time)

```bash
cargo install cargo-instruments
```

Requires Xcode installed. Verify:

```bash
cargo instruments --list-templates
```

## Profile infer (hot path)

Run from workspace root. Use absolute paths for data files:

```bash
cd /path/to/spma  # workspace root — Cargo.toml is here

cargo instruments -p spma-cli --release --template "Time Profiler" -- infer \
  --model /path/to/spma-experiments/hdfs-validation/data/model/corpus/hdfs_50000.json \
  --input /path/to/spma-experiments/hdfs-validation/data/splits/test_normal.txt > /dev/null
```

Instruments.app opens automatically. In the call tree:

1. Filter to `spma` binary
2. Sort by "Self Weight" descending
3. Record top 5 functions by % CPU

## Profile train (sequential path)

Run from workspace root:

```bash
cd /path/to/spma

cargo instruments -p spma-cli --release --template "Time Profiler" -- train \
  --corpus <(head -50000 /path/to/spma-experiments/hdfs-validation/data/splits/train_normal.txt) \
  --output /tmp/hdfs_50000_prof.json \
  --beam 10
```

## What to look for

Suspected hotspots (unconfirmed — this is what profiling will validate):

- `beam_search` inner loop — candidate expansion
- `PartialAlignment::clone` — called per candidate per symbol
- candidate sort (`Vec::sort_by`) — O(k log k) per symbol position
- `can_extend` — HashMap cursor lookups

## Recording results

Update `docs/performance.md` under `## Current implementation` with:

```
Profiled on HDFS 50k train / 446k infer, Apple Silicon, release build:
- Top hotspot: <function> — X% self CPU
- Second: <function> — Y%
- ...
```

Then pick the next optimization from `## Potential improvements` based on which
hotspot dominates.

## Symbols / debug info

`cargo instruments` with `--release` uses the release profile. For symbol
resolution add to workspace `Cargo.toml` (root, not crate-level):

```toml
[profile.release]
debug = 1
```

**Revert after profiling** — increases binary size and link time.

## Known findings (HDFS 446k infer, 2026-07-18/19)

Inverted call tree top consumers at baseline:

| Self time | Symbol | Root cause |
|-----------|--------|------------|
| ~80s | hashbrown (find_inner, SipHasher, hash_one…) | `old_cursors`/`new_cursors: HashMap<usize,usize>` in `PartialAlignment` cloned per candidate per symbol |
| ~25s | `<f64>::log2` + dyld stub | MDL cost recomputed per candidate per extension |

Optimization iterations (each re-profiled):

| Profile pass | Top consumer | Outcome |
|---|---|---|
| Baseline | 40.5% `Map::fold`, 35.9% `hash_one` | — |
| After H (Vec cursors + bitmask) | 81.2% `Vec::from_iter` | HashMap gone; pid_freq alloc now dominant |
| After pid_freq Vec | 88.3% `LocalKey::with` | TLS overhead exceeded benefit — reverted |
| After grow-only Vec buffers | 83.7% `infer`, 8.5% `log2` | No single bottleneck — healthy profile |

Final result: ~637s user / ~42s wall vs ~1816s / ~123s baseline (~2.9× wall speedup).

H ✓, I ✓ implemented. J (TLS) attempted and reverted. See `docs/performance.md` for full details.
