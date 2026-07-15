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

## CLI usage

### Train

```bash
spma train normal_sequences.txt
# saves ./spma_grammar.bin
```

```bash
spma train normal_sequences.txt --grammar /path/to/custom.bin
spma train --verbose normal_sequences.txt   # print alignment tables during training
```

### Infer

```bash
spma infer test_input.txt
# loads ./spma_grammar.bin, prints OK/ANOMALY per line
# exits 1 if any anomaly detected (E > 0)
```

```bash
spma infer test_input.txt --grammar /path/to/custom.bin
spma infer --verbose test_input.txt   # print full alignment tables
```

### Input format

One sequence per line, space-separated symbols. Lines starting with `#` are ignored.

```
TRIP_A BREAKER_OPEN UNDERVOLTAGE BACKUP_RELAY
TRIP_B BREAKER_OPEN OVERCURRENT BACKUP_RELAY
# this line is a comment
```

Special symbol prefixes:

| Prefix | Type | Example |
|---|---|---|
| `<` / `>` | Boundary markers | `< event >` |
| `#` | Unique ID symbol | `#session_42` |
| `!` | Identification symbol | `!admin` |
| _(none)_ | Data symbol (default) | `BREAKER_OPEN` |

## Library usage

```rust
use spma::Spma;

let mut engine = Spma::new();
engine.train(&[
    vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE"],
    vec!["TRIP_A", "BREAKER_OPEN", "OVERCURRENT"],
])?;
engine.save("grammar.bin")?;

let engine = Spma::load("grammar.bin")?;
let result = engine.infer(&["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE"])?;

println!("E={:.3}  CD={:+.3}  anomaly={}", result.e_cost, result.cd, result.is_anomaly);
println!("{}", result.alignment);
```

`InferResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `e_cost` | `f64` | Encoding cost of unmatched symbols (E in T=G+E) |
| `cd` | `f64` | Compression difference; positive = grammar compresses the sequence |
| `is_anomaly` | `bool` | `true` when `e_cost > 0` |
| `unmatched` | `Vec<String>` | Symbol names not covered by any grammar pattern |
| `alignment` | `String` | Human-readable alignment table |

## Examples

```bash
cargo run --example fault_detection
```

`examples/fault_detection.rs` — trains on normal industrial fault sequences, saves grammar, loads it, then classifies a set of test inputs including unknown fault types and novel event streams.

## Use cases

Industrial log anomaly detection (learn normal sequences, flag high-E inputs), protocol conformance checking (align captured traffic against spec patterns), and fault code classification (one grammar per class, pick minimum T). In all cases the alignment table is the explanation — no post-hoc attribution.

**Known limitation**: beam search matches symbols by identity, not position — order violations are not detected unless the grammar contains ordered multi-symbol patterns. Use boundary markers (`<` / `>`) to give the grammar positional anchors.

## Features

| Feature | Location | Notes |
|---|---|---|
| String interning (symbol → u32 ID) | `src/intern.rs` | O(1) symbol comparison |
| Shannon bit costs | `src/engine.rs` | `-log2(freq/total)`, no distortion |
| T=G+E scoring | `src/lib.rs` | G charged once at insertion |
| Staged beam search (SPMA core) | `src/beam.rs` | Monotonicity constraint enforced |
| Learning loop with MDL gate | `src/engine.rs` | n-gram bootstrap + beam-driven extraction |
| One-trial learning | `src/engine.rs` | Add-only store, no forgetting |
| Alignment table printer | `src/engine.rs` | Per-symbol coverage display |
| Unknown symbol detection | `src/lib.rs` | Symbols absent from training → forced E > 0 |
| Grammar persistence | `src/lib.rs` | serde + bincode |

## Architecture

```
CLI (main.rs)
       ↓
Spma API (lib.rs)
       ↓
Learning loop (engine.rs)
       ↓
Beam search (beam.rs)
       ↓
T=G+E scoring (model.rs)
       ↓
String interning: symbol → u32 (intern.rs)
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
- Order violations undetected — grammar converges to single-symbol patterns; beam search is order-agnostic (see [docs/known-issues.md](docs/known-issues.md))
- Performance cap ~1k patterns before Phases B–F (inverted index, parallel beam) are implemented

## References

- J G Wolff, SP Theory: https://www.cognitionresearch.org/sp.htm
