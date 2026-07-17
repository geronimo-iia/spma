# Task: Optimize `spma infer` throughput

## Context

`spma infer` on 111k HDFS sequences takes ~10 min (estimated from 16k anomaly run).
Each sequence is independent — pure read-only on a shared `Spma` model.
Current implementation: sequential `for line in buf.lines()` loop in `src/bin/spma.rs`.

Codebase: `/Users/geronimo/dev/projects/libraries/spma/`
File to change: `src/bin/spma.rs` (infer branch only)

## Fix 1 — Buffered stdout

Current code calls `println!()` per line — one syscall per sequence.
With 111k sequences that's 111k flushes.

Wrap stdout in `BufWriter` and flush once at end:

```rust
use std::io::Write;

let stdout = std::io::stdout();
let mut out = std::io::BufWriter::new(stdout.lock());

// replace println!(...) with:
writeln!(out, "...")?;

// BufWriter flushes automatically on drop, but be explicit:
out.flush()?;
```

Apply to both the `--json` and plain-text output paths.

## Fix 2 — Parallel inference with rayon

Read all input lines into a `Vec<String>` first, then process in parallel,
then write results in original order.

Add `rayon` to `Cargo.toml` if not already present:

```toml
[dependencies]
rayon = "1"
```

Infer branch rewrite:

```rust
use rayon::prelude::*;

// 1. Collect all non-empty lines
let lines: Vec<String> = buf
    .lines()
    .filter_map(|l| l.ok())
    .map(|l| l.trim().to_owned())
    .filter(|l| !l.is_empty())
    .collect();

// 2. Run inference in parallel — Spma is read-only, safe to share
let results: Vec<(Vec<String>, spma::engine::InferResult)> = lines
    .par_iter()
    .map(|line| {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let result = spma.infer(&tokens);
        (tokens.iter().map(|s| s.to_string()).collect(), result)
    })
    .collect();

// 3. Write output in order (single thread, buffered)
let stdout = std::io::stdout();
let mut out = std::io::BufWriter::new(stdout.lock());
let mut any_anomaly = false;

for (tokens, result) in &results {
    if result.is_anomaly {
        any_anomaly = true;
    }
    if json {
        writeln!(out,
            "{{\"seq\":{:?},\"e_cost\":{:.6},\"e_norm\":{:.6},\"cd\":{:.6},\"anomaly_percentile\":{:.6},\"is_anomaly\":{}}}",
            tokens, result.e_cost, result.e_norm, result.cd, result.anomaly_percentile, result.is_anomaly,
        )?;
    } else {
        writeln!(out,
            "{}\te_norm={:.4}\tpct={:.4}\t{}",
            tokens.join(" "),
            result.e_norm,
            result.anomaly_percentile,
            if result.is_anomaly { "ANOMALY" } else { "ok" },
        )?;
    }
}
out.flush()?;
```

**Thread safety**: `Spma::infer` takes `&self` (immutable). Verify it does not
mutate any interior state — if it does, wrap with `Arc<Mutex<>>` or fix the
mutability. Do not guess — read the signature in `src/engine.rs` before
proceeding.

## Fix 3 — Avoid per-sequence token allocation (minor)

Each `line.split_whitespace().collect::<Vec<&str>>()` borrows from the line
string. In the rayon closure the line is already a `&String` so this works
naturally. No extra allocation needed — just confirm the borrow lifetimes are
correct and do not introduce unnecessary `.to_string()` copies.

## Constraints

- Output order must be preserved (rayon `par_iter().collect()` guarantees this)
- Exit code 1 on any anomaly must be preserved
- `--json` and plain-text paths both parallelized
- No changes to `src/engine.rs`, `src/lib.rs`, or test files
- `cargo test` must pass
- `cargo clippy` must pass

## Measure

```bash
# Before
time /path/to/spma infer \
  --model /tmp/hdfs_50k.json \
  --input data/test_normal.txt \
  --json > /dev/null

# After
time /path/to/spma infer \
  --model /tmp/hdfs_50k.json \
  --input data/test_normal.txt \
  --json > /dev/null
```

Target: 8–10× speedup on 8-core machine (bound by `Spma::infer` per-seq cost).

Correctness check — output must be bit-identical:

```bash
# Save reference before change
/path/to/spma_before infer --model /tmp/hdfs_50k.json \
  --input data/test_anomaly.txt --json > /tmp/ref.jsonl

# After change
/path/to/spma_after infer --model /tmp/hdfs_50k.json \
  --input data/test_anomaly.txt --json > /tmp/new.jsonl

diff /tmp/ref.jsonl /tmp/new.jsonl  # must be empty
```

## Commit

```
perf(cli): parallelize infer with rayon, buffer stdout output
```
