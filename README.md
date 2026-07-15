# spma

Rust implementation of **SP Multiple Alignment (SPMA)** — a transparent pattern-matching engine for discrete sequential data.

SPMA originates from J G Wolff's SP Theory of Intelligence (1987–2006), a unifying framework grounded in the principle that cognition reduces to compression. The core operation is multiple alignment: matching a New pattern simultaneously against a grammar of Old patterns, minimising description length T = G + E. This implementation applies that mechanism narrowly to structured sequential data — no cognitive modeling, no numeric signals.

**Status**: exploratory — built to understand whether SPMA is practically viable for anomaly detection. Not production-ready, not benchmarked on real data.

**Scope**: correct, auditable anomaly detection on event sequences. Not a general intelligence system. Not a competitor to neural approaches.

## What it does

Learns compressed grammars from sequential data, then aligns new inputs against the learned grammar:

- `E > 0` → anomaly score (unmatched symbols cost bits)
- Alignment table → localization (which symbols broke the pattern)
- No post-hoc attribution needed — the alignment IS the explanation

Scoring objective: **T = G + E** (MDL). G charged once at insertion, E = bit cost of unmatched New symbols.

## Usage

### Train

```bash
cargo run --release -- normal_sequences.txt
# saves ./spma_grammar.bin
```

### Inference

```bash
cargo run --release -- --no-learn test_input.txt
# loads ./spma_grammar.bin, prints alignment table per line
# [UNMATCHED:symbol] for unknown symbols, E > 0 triggers ANOMALY DETECTED banner
```

### Custom grammar path

```bash
cargo run --release -- --no-learn --grammar /path/to/custom.bin test_input.txt
```

## Input format

One pattern per line, space-separated symbols:

```
TRIP_A BREAKER_OPEN UNDERVOLTAGE BACKUP_RELAY
FAULT_B OVERCURRENT TRIP_B
normal_start normal_op normal_end
```

## Use cases

Industrial log anomaly detection (learn normal sequences, flag high-E inputs), protocol conformance checking (align captured traffic against spec patterns), and fault code classification (one grammar per class, pick minimum T). In all cases the alignment table is the explanation — no post-hoc attribution.

## Features

| Feature | Location | Notes |
|---|---|---|
| String interning (symbol → u32 ID) | `src/intern.rs` | O(1) symbol comparison |
| Shannon bit costs | `src/engine.rs` | `-log2(freq/total)`, no distortion |
| T=G+E scoring | `src/lib.rs` | G charged once at insertion |
| Staged beam search (SPMA core) | `src/beam.rs` | Monotonicity constraint enforced |
| Learning loop with MDL gate | `src/engine.rs` | n-gram bootstrap + beam-driven extraction |
| One-trial learning | `src/engine.rs` | Add-only store, no forgetting |
| Alignment table printer | `src/engine.rs` | `[UNMATCHED:x]` for unknown symbols |
| `--no-learn` inference mode | `src/main.rs` | Essential for anomaly detection |
| Grammar persistence | `src/main.rs` | serde + bincode, `--grammar` flag |

## Architecture

```
CLI (main.rs)
       ↓
Learning loop (engine.rs)
       ↓
Beam search (beam.rs)
       ↓
T=G+E scoring (lib.rs)
       ↓
String interning: symbol → u32 (intern.rs)
       ↓
Data model: Pattern, Symbol, Alignment, Grammar (lib.rs)
```

Key decisions — see [docs/architecture.md](docs/architecture.md):
- G charged once at insertion (not per alignment — otherwise CD ≈ 0, learning signal vanishes)
- Monotonicity constraint in beam search (Old patterns advance forward only)
- Multi-pattern simultaneous alignment (not pairwise — cross-pattern coverage drives compression gain)
- Shannon bit costs: `cost(s) = -log2(freq/total)`, no tunable distortion factor


## Docs

| File | Content |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Scoring rationale, beam search, learning loop |
| [docs/performance.md](docs/performance.md) | Possible improvements (only Phase A implemented) |


## Limitations

- No numeric representation (Wolff acknowledged this gap; no solution exists)
- No 2D patterns, probabilistic inference, or cognitive modeling
- Not validated on real data yet
- Performance cap ~1k patterns before Phases B–F (inverted index, parallel beam) are implemented

## References

- J G Wolff, SP Theory: https://www.cognitionresearch.org/sp.htm
