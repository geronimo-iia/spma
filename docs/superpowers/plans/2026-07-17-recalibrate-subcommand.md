# Add `spma recalibrate` Subcommand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `recalibrate` CLI subcommand that freezes grammar structure and replays a corpus to refit `EDistribution` only.

**Architecture:** Add `pub fn recalibrate(&mut self, corpus: &[Vec<&str>])` to `Spma` in `engine.rs` — it runs `infer()` on each sequence and collects `e_norm`/`level_e_norms`, then calls `EDistribution::fit()`. Add a `Recalibrate` variant to the `Command` enum in `spma.rs` that loads model, reads corpus, calls recalibrate, applies optional threshold override, and saves.

**Tech Stack:** Rust, clap (existing), serde_json (existing), anyhow (existing)

---

### Task 1: Add `recalibrate` method to `engine.rs`

**Files:**
- Modify: `src/engine.rs` (after `infer` method, around line 639)

- [ ] **Step 1: Write the failing test in `engine.rs` under `#[cfg(test)]`**

Add at the end of the `tests` module (after `test_serialization_roundtrip`, around line 1196):

```rust
#[test]
fn recalibrate_does_not_change_grammar_structure() {
    let seq = vec!["A", "B", "C", "A", "B", "C"];
    let corpus = make_corpus(seq.clone(), 6);
    let mut spma = Spma::new(5);
    spma.train(&corpus);

    let levels_before: Vec<usize> = spma.grammar.levels.iter().map(|l| l.patterns.len()).collect();
    let atom_costs_before = spma.atom_costs.clone();
    let interner_len_before = spma.grammar.interner.len();

    let corpus_refs: Vec<Vec<&str>> = corpus.iter().map(|s| s.iter().copied().collect()).collect();
    spma.recalibrate(&corpus_refs);

    let levels_after: Vec<usize> = spma.grammar.levels.iter().map(|l| l.patterns.len()).collect();
    assert_eq!(levels_before, levels_after, "grammar levels must not change");
    assert_eq!(atom_costs_before, spma.atom_costs, "atom_costs must not change");
    assert_eq!(interner_len_before, spma.grammar.interner.len(), "interner must not change");

    // e_distribution must be populated (not all zeros)
    assert!(
        !spma.grammar.e_distribution.sorted_e_norms_len_for_test() == 0,
        "e_distribution must be populated after recalibrate"
    );
}

#[test]
fn recalibrate_after_pattern_removal() {
    let seq = vec!["A", "B", "C", "A", "B", "C"];
    let corpus = make_corpus(seq.clone(), 6);
    let mut spma = Spma::new(5);
    spma.train(&corpus);

    // Remove one pattern from level 0
    let original_count = spma.grammar.levels[0].patterns.len();
    if original_count > 1 {
        spma.grammar.levels[0].patterns.pop();
        let reduced_count = spma.grammar.levels[0].patterns.len();
        assert_eq!(reduced_count, original_count - 1);

        let corpus_refs: Vec<Vec<&str>> = corpus.iter().map(|s| s.iter().copied().collect()).collect();
        spma.recalibrate(&corpus_refs);

        assert_eq!(
            spma.grammar.levels[0].patterns.len(),
            reduced_count,
            "recalibrate must not re-add removed patterns"
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail (method doesn't exist yet)**

```bash
cd /Users/geronimo/dev/projects/libraries/spma && cargo test recalibrate 2>&1 | head -30
```

Expected: compile error — `no method named 'recalibrate'` and `sorted_e_norms_len_for_test` not found.

- [ ] **Step 3: Add test helper to `EDistribution` in `model.rs`**

The test needs to inspect `sorted_e_norms` (private). Add a `#[cfg(test)]` accessor.

Find the `impl EDistribution` block in `src/model.rs` and add after the existing methods:

```rust
#[cfg(test)]
pub fn sorted_e_norms_len_for_test(&self) -> usize {
    self.sorted_e_norms.len()
}
```

- [ ] **Step 4: Fix test assertion (typo in Step 1)**

The assertion `!spma.grammar.e_distribution.sorted_e_norms_len_for_test() == 0` has an accidental `!`. Replace with:

```rust
assert!(
    spma.grammar.e_distribution.sorted_e_norms_len_for_test() > 0,
    "e_distribution must be populated after recalibrate"
);
```

(Edit the test you wrote in Step 1 to fix this.)

- [ ] **Step 5: Add `recalibrate` method to `Spma` in `engine.rs`**

Add after the closing brace of `pub fn infer(...)` (around line 639), still inside `impl Spma`:

```rust
pub fn recalibrate(&mut self, corpus: &[Vec<&str>]) {
    let mut e_norms: Vec<f64> = Vec::with_capacity(corpus.len());
    let mut level_e_norms_vecs: Vec<Vec<f64>> = Vec::new();

    for seq in corpus {
        let result = self.infer(seq);
        e_norms.push(result.e_norm);
        for (i, &lvl_e) in result.level_e_norms.iter().enumerate() {
            if level_e_norms_vecs.len() <= i {
                level_e_norms_vecs.push(Vec::new());
            }
            level_e_norms_vecs[i].push(lvl_e);
        }
    }

    self.grammar.e_distribution =
        crate::model::EDistribution::fit(e_norms, 0.0, level_e_norms_vecs);
}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /Users/geronimo/dev/projects/libraries/spma && cargo test recalibrate 2>&1
```

Expected output:
```
test tests::recalibrate_does_not_change_grammar_structure ... ok
test tests::recalibrate_after_pattern_removal ... ok
```

- [ ] **Step 7: Run all tests to confirm no regressions**

```bash
cd /Users/geronimo/dev/projects/libraries/spma && cargo test 2>&1 | tail -20
```

Expected: all tests pass, no failures.

- [ ] **Step 8: Commit**

```bash
cd /Users/geronimo/dev/projects/libraries/spma
git add src/engine.rs src/model.rs
git commit -m "feat(engine): add recalibrate method to Spma"
```

---

### Task 2: Add `Recalibrate` subcommand to `spma.rs`

**Files:**
- Modify: `src/bin/spma.rs`

- [ ] **Step 1: Add `Recalibrate` variant to `Command` enum**

In `src/bin/spma.rs`, find the `Command` enum (around line 20). Add after the `Infer { ... },` variant, before the closing `}`:

```rust
/// Reload a model, replay corpus to refit e_distribution, save updated model
Recalibrate {
    /// Path to saved model (modified in place or written to --output)
    #[arg(short, long)]
    model: String,

    /// Training corpus to replay (one sequence per line, tokens space-separated)
    #[arg(short, long)]
    corpus: String,

    /// Output path; if omitted, overwrites --model
    #[arg(short, long)]
    output: Option<String>,

    /// Override anomaly threshold after recalibration
    #[arg(short, long)]
    threshold: Option<f64>,
},
```

- [ ] **Step 2: Add handler in `main()`**

In `main()`, find the `match cli.command {` block. Add a new arm after the `Command::Infer { ... }` arm, before the closing `}` of the match:

```rust
Command::Recalibrate {
    model,
    corpus,
    output,
    threshold,
} => {
    let f = File::open(&model).with_context(|| format!("open model: {model}"))?;
    let mut spma =
        Spma::load(BufReader::new(f)).with_context(|| format!("load model: {model}"))?;

    let raw = read_corpus(&corpus)?;
    let corpus_refs: Vec<Vec<&str>> = raw
        .iter()
        .map(|seq| seq.iter().map(String::as_str).collect())
        .collect();

    spma.recalibrate(&corpus_refs);

    if let Some(t) = threshold {
        spma.set_anomaly_threshold(t);
    }

    let out_path = output.as_deref().unwrap_or(&model);
    let f = File::create(out_path).with_context(|| format!("create output: {out_path}"))?;
    spma.save(BufWriter::new(f))
        .with_context(|| format!("save model: {out_path}"))?;

    let dist = spma.e_distribution();
    eprintln!(
        "recalibrated: {} sequences, threshold={:.4}",
        raw.len(),
        dist.threshold
    );
}
```

- [ ] **Step 3: Verify it compiles**

```bash
cd /Users/geronimo/dev/projects/libraries/spma && cargo build 2>&1 | head -30
```

Expected: compiles without errors.

- [ ] **Step 4: Run all tests**

```bash
cd /Users/geronimo/dev/projects/libraries/spma && cargo test 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/geronimo/dev/projects/libraries/spma
git add src/bin/spma.rs
git commit -m "feat(cli): add recalibrate subcommand"
```

---

### Task 3: Build release and verify round-trip

**Files:** none (verification only)

- [ ] **Step 1: Build release binary**

```bash
cd /Users/geronimo/dev/projects/libraries/spma && cargo build --release 2>&1 | tail -5
```

Expected: `Finished release` with no errors.

- [ ] **Step 2: Verify CLI help shows recalibrate**

```bash
/Users/geronimo/dev/projects/libraries/spma/target/release/spma --help
/Users/geronimo/dev/projects/libraries/spma/target/release/spma recalibrate --help
```

Expected: `recalibrate` listed in subcommands; `--model`, `--corpus`, `--output`, `--threshold` args visible.

- [ ] **Step 3: Confirm decision rule — round-trip threshold within 1%**

If `hdfs-validation/data/model/hdfs_base.json` and `hdfs-validation/data/splits/train_normal.txt` exist, run:

```bash
cd /Users/geronimo/dev/projects/libraries/spma

# Capture original threshold
ORIG=$(./target/release/spma grammar --model hdfs-validation/data/model/hdfs_base.json --json | python3 -c "import sys,json; print(json.load(sys.stdin)['threshold'])")

# Recalibrate on same training corpus
./target/release/spma recalibrate \
    --model hdfs-validation/data/model/hdfs_base.json \
    --corpus hdfs-validation/data/splits/train_normal.txt \
    --output /tmp/hdfs_recalibrated.json

# Capture new threshold
NEW=$(./target/release/spma grammar --model /tmp/hdfs_recalibrated.json --json | python3 -c "import sys,json; print(json.load(sys.stdin)['threshold'])")

echo "orig=$ORIG new=$NEW"
python3 -c "o,n=float('$ORIG'),float('$NEW'); diff=abs(o-n)/max(abs(o),1e-12); print(f'diff={diff:.4%}'); exit(0 if diff < 0.01 else 1)"
```

Expected: `diff=X.XX%` where X.XX < 1.00. If test data doesn't exist, skip and note.

- [ ] **Step 4: Confirm grammar structure unchanged**

```bash
cd /Users/geronimo/dev/projects/libraries/spma
./target/release/spma grammar --model hdfs-validation/data/model/hdfs_base.json --json | python3 -c "import sys,json; d=json.load(sys.stdin); print('levels:', len(d['levels'])); [print(f'  level {l[\"level\"]}: {l[\"pattern_count\"]} patterns') for l in d['levels']]"
./target/release/spma grammar --model /tmp/hdfs_recalibrated.json --json | python3 -c "import sys,json; d=json.load(sys.stdin); print('levels:', len(d['levels'])); [print(f'  level {l[\"level\"]}: {l[\"pattern_count\"]} patterns') for l in d['levels']]"
```

Expected: identical level counts and pattern counts in both outputs.
