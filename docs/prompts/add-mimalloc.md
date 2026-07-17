# Task: Try mimalloc as global allocator

## Context

SPMA training and inference both use rayon for parallel beam_search, producing
many small concurrent allocations (Vec candidates, match logs, pid sequences).
macOS default allocator has thread contention under parallel load. mimalloc
is a drop-in replacement with better parallel throughput.

This is a one-shot measurement task. If speedup < 5%, revert and close.

## Change

`Cargo.toml`:

```toml
[dependencies]
mimalloc = { version = "0.1", default-features = false }
```

`src/bin/spma.rs` — top of file:

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

No other changes.

## Measure

Train benchmark (1k sequences):

```bash
# baseline (already measured: ~3.5s after rayon)
time /path/to/spma_before train \
  --corpus /tmp/hdfs_1k.txt \
  --output /tmp/hdfs_1k.json \
  --beam 10

# with mimalloc
cargo build --release
time ./target/release/spma train \
  --corpus /tmp/hdfs_1k.txt \
  --output /tmp/hdfs_1k.json \
  --beam 10
```

Infer benchmark (111k sequences):

```bash
time ./target/release/spma infer \
  --model /tmp/hdfs_50k.json \
  --input data/test_normal.txt \
  --json > /dev/null
```

Run each 3 times, take median.

## Decision rule

| Speedup | Action |
|---|---|
| < 5% | Revert — remove dependency and global_allocator line |
| 5–15% | Keep, note in commit message |
| > 15% | Keep, worth documenting in README |

## Correctness

```bash
diff <(./target/release/spma infer --model /tmp/hdfs_50k.json --input data/test_anomaly.txt --json) \
     /tmp/ref_anomaly.jsonl
```

Must be empty.

## Commit (if kept)

```
perf(cli): use mimalloc global allocator for parallel allocation throughput
```

## Revert commit (if not kept)

```
chore: remove mimalloc trial — speedup < 5%, not worth dependency
```
