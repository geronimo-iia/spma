# Task: Add `--progress` flag to `spma train`

## Goal

Add an optional `--progress` flag to the `spma train` CLI subcommand that prints
periodic progress lines to stderr during training. Off by default — zero overhead
when not requested.

## Scope

Two files only:
- `src/bin/spma.rs` — add flag, pass to engine or drive progress loop
- `src/engine.rs` — expose a callback or iterator hook on `train`

Do NOT touch `src/lib.rs`, tests, or any other file.

## What to print (stderr)

One line per reported batch, format:

```
[spma] trained 10000/446579 (2.2%) | grammar: 9 levels, 42 patterns | elapsed: 3.1s
```

Fields:
- `trained N/total` — sequences processed so far / total
- `(pct%)` — percentage, 1 decimal
- `grammar: L levels, P patterns` — current `grammar.levels.len()` and total pattern
  count across all levels
- `elapsed: T s` — wall-clock seconds since train started (1 decimal)

Report every 10 000 sequences (configurable constant `PROGRESS_INTERVAL = 10_000`
at top of `spma.rs`). Also print one final line when training completes:

```
[spma] done: 446579 sequences | grammar: 9 levels, 42 patterns | elapsed: 37.4s
```

## CLI change (`src/bin/spma.rs`)

Add to `Command::Train`:

```rust
/// Print progress to stderr every 10 000 sequences
#[arg(long, default_value_t = false)]
progress: bool,
```

Pass `progress` into the engine call. Do not break the existing `--beam`,
`--max-gap`, `--threshold` flags.

## Engine change (`src/engine.rs`)

Current signature:

```rust
pub fn train(&mut self, corpus: &[Vec<&str>])
```

Add an optional progress callback:

```rust
pub fn train_with_progress<F>(&mut self, corpus: &[Vec<&str>], mut on_progress: F)
where
    F: FnMut(usize),   // called with index of last processed sequence (0-based)
```

`train_with_progress` is identical to `train` but calls `on_progress(i)` after
each sequence alignment in the inner loop (Step 4, the `for (i, seq)` loop).

Keep `train` as a thin wrapper:

```rust
pub fn fn train(&mut self, corpus: &[Vec<&str>]) {
    self.train_with_progress(corpus, |_| {});
}
```

The callback only fires on the inner per-sequence loop (not Step 3 cold-start),
because cold-start runs fast compared to the N-level outer loop.

## In `spma.rs` — progress printer

```rust
let total = raw.len();
let start = std::time::Instant::now();

if progress {
    spma.train_with_progress(&corpus_refs, |i| {
        let done = i + 1;
        if done % PROGRESS_INTERVAL == 0 || done == total {
            let pct = done as f64 / total as f64 * 100.0;
            let levels = spma.grammar.levels.len();
            let patterns: usize = spma.grammar.levels.iter().map(|l| l.patterns.len()).sum();
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "[spma] trained {done}/{total} ({pct:.1}%) | grammar: {levels} levels, {patterns} patterns | elapsed: {elapsed:.1}s"
            );
        }
    });
} else {
    spma.train(&corpus_refs);
}
```

## Existing output unchanged

The final `eprintln!("trained: {} sequences...")` line that already exists after
training must stay. Progress lines are additive.

## Verify

```bash
head -5000 data/train_normal.txt > /tmp/hdfs_5k.txt
cargo run --release --bin spma -- train \
  --corpus /tmp/hdfs_5k.txt \
  --output /tmp/hdfs_5k.json \
  --progress
```

Expected: progress lines appear every 1000 sequences (adjust `PROGRESS_INTERVAL`
to 1000 for this small test, or just check first line appears at 10000 if corpus
>10k).

Also verify without `--progress`:

```bash
cargo run --release --bin spma -- train \
  --corpus /tmp/hdfs_5k.txt \
  --output /tmp/hdfs_5k.json
```

No progress lines, only final `trained:` line on stderr.

## Commit

```
feat(cli): add --progress flag to spma train for periodic stderr output
```
