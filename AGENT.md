# AGENT.md — Working in the spma codebase

Guidance for agentic workers. Read this before touching any file.

---

## Workspace layout

```
spma/                        ← workspace root (run cargo/git from here)
  spma/src/                  ← library crate
    beam.rs                  ← beam search hot path, PartialAlignment, MatchArena
    engine.rs                ← Spma, train, infer, recalibrate, retrain, MDL loop
    model.rs                 ← Grammar, GrammarLevel, Pattern, EDistribution, SymbolIndex
    alignment.rs             ← Alignment, AlignmentRow, build_alignment, Display
    intern.rs                ← Interner (str → u32)
    lib.rs                   ← public re-exports only
  spma/tests/                ← integration tests (cargo test --tests)
  spma-cli/src/main.rs       ← CLI binary (train/infer/retrain/recalibrate/grammar)
  docs/
    architecture.md          ← scoring objective, beam algorithm, learning loop
    grammar-spec.md          ← data model, what was excluded and why
    scoring.md               ← E_norm, threshold semantics, per-level calibration
    performance.md           ← profiling results, applied optimizations, remaining work
    known-limitations.md     ← reordering detection, F1 ceiling
    superpowers/plans/       ← implementation plans (checkbox format)
```

The library crate is `spma/` (subdirectory), not the workspace root. All
`cargo` commands that target the library use `-p spma`. The CLI is `-p spma-cli`.

---

## How to verify your work

```bash
# All library tests (unit + integration)
cargo test -p spma 2>&1 | tail -20

# Full workspace (includes CLI compile check)
cargo test 2>&1 | tail -20

# Clippy — must pass with zero warnings
cargo clippy -p spma -- -D warnings 2>&1 | tail -20

# Release build (required before any performance claim)
cargo build --release 2>&1 | grep "^error"
```

All tests must pass before committing. Zero clippy warnings is a hard
requirement — the workspace has `[lints.clippy] all = "warn"`.

---

## Module dependency order

```
model.rs  ←  intern.rs
beam.rs   ←  model.rs
alignment.rs ← beam.rs, model.rs
engine.rs ← beam.rs, model.rs, alignment.rs
lib.rs    ← engine.rs, model.rs, intern.rs
```

`beam.rs` must NOT import from `engine.rs` — circular. If a constant or type
needs to be shared between them, define it in `beam.rs` and re-export from
`engine.rs`. See `MAX_BITMASK_SYMBOLS` as the established pattern.

---

## Hard constraints — do not violate

**Sequence length limit**: `beam_search` asserts `new_len <= MAX_BITMASK_SYMBOLS`
(512). This is a hard limit from the `[u64; 8]` bitmask in `PartialAlignment`.
Any code path that calls `beam_search` must validate input length first.
`validate_corpus` / `validate_sequence` in `engine.rs` are the canonical guards.

**`PartialAlignment` clone must stay allocation-free**: the beam inner loop
clones `PartialAlignment` per candidate per symbol. Any heap allocation in
clone kills performance. Current layout:
- `new_cursors: [u16; MAX_PATS]` — fixed array, stack copy
- `covered_new: [u64; 8]` — fixed array, stack copy
- `log_tail: Option<u32>` — stack copy
Do not add `Vec`, `HashMap`, `String`, or `Box` fields to `PartialAlignment`.

**`SymbolIndex` is not serialized**: `#[serde(skip)]` on `GrammarLevel::symbol_index`.
It is rebuilt in `Spma::load` and after every `patterns.extend(...)` call via
`rebuild_index()`. If you add a code path that mutates `patterns`, you must call
`rebuild_index()` afterward or `beam_search` will use a stale index.

**`old_cursors` was removed**: the field was written but never read. Do not
re-add it. `can_extend` uses only `new_cursors`.

**`train` is a cold start**: it resets grammar, atom_freq, and atom_costs.
`retrain` is incremental — it calls `train_inner` without resetting. Do not
conflate them.

**`recalibrate` must preserve `threshold`**: `recalibrate` saves and restores
`e_distribution.threshold` and `level_thresholds` across the refit. Follow the
same save/restore pattern if you add any code that calls `EDistribution::fit`.

---

## Sharp edges that cause silent wrong behavior

**`GAP_MARKER = u32::MAX`**: used as a sentinel in ngram encoding
`[sym_i, GAP_MARKER, gap_size, sym_j]`. If any real symbol ID reaches
`u32::MAX`, it will be misinterpreted as a gap marker. The interner asserts
`names.len() < u32::MAX` — do not remove that assert.

**`u16::MAX` cursor sentinel**: `new_cursors[old_idx] == u16::MAX` means
"pattern not yet started". A `new_pos` of 65535 would be misread as absent.
The `debug_assert!(new_pos < u16::MAX as usize)` in `extend_match` guards this.
Do not remove it.

**`build_alignment` takes `old_patterns: &[&Pattern]`**: `old_idx` in
`MatchEvent` is a beam-slice index, not a grammar pattern ID. The slice may
contain candidate patterns not yet in the grammar. Do not change this to a
grammar lookup.

**`compute_total_e_dp` is O(n × m) per sentence**: intentional for correctness
during MDL gating. Do not replace with beam search — the DP gives the exact
minimum encoding cost, which beam search does not guarantee.

---

## Plans

Plans live in `docs/superpowers/plans/` as Markdown files with checkbox steps.
When a plan is fully implemented, delete the file — completed plans belong in
git history, not the working tree.

Before executing a plan, verify:
1. All file paths in the plan match the actual workspace layout (`spma/src/`,
   not `src/` or `spma/spma/src/`)
2. `git add` paths are relative to workspace root (`spma/src/beam.rs`, not
   `src/beam.rs`)
3. No step imports from a module that depends on the importing module

---

## Commit conventions

```
feat(scope): short description
fix(scope): short description
perf(scope): short description
test(scope): short description
docs(scope): short description
refactor(scope): short description
```

Scope is the primary file or subsystem: `beam`, `engine`, `model`, `alignment`,
`cli`, `intern`. One logical change per commit. Tests for a change go in the
same commit as the change.

---

## What is intentionally deferred to v0.2

Do not implement these without an explicit plan:

- Structure-of-Arrays layout for `PatternStore` (perf item B)
- Parallel training (perf item D)
- SIMD symbol comparison (perf item E)
- `infer_internal` fast path (no match log) — removed, not missed
- Examples `fault_detection.rs`, `ordered_sequences.rs` — compile but output
  may not match doc comments exactly; do not update doc comments to match
  observed output without running the example first

---

## Key numbers to know

- HDFS benchmark: F1 = 0.893 unsupervised (50k train, 446k infer)
- Infer throughput after H+I: ~2.9× wall speedup vs baseline (42s vs 123s wall,
  16 cores, Apple Silicon)
- Beam default: `beam_k = 10`
- Max induced gap default: `MAX_INDUCED_GAP = 3`
- Bitmask limit: `MAX_BITMASK_SYMBOLS = 512`
- Cursor array limit: `MAX_PATS = 128` patterns per beam call
- Max grammar levels: 8 (hardcoded in `train_inner`)
