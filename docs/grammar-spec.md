# Grammar Specification — SPMA Full SP Theory Model

Design document for the grammar data model, serialization format, and extension points
required to implement the full SP theory feature set. This doc comes before any
implementation. Change the spec here first; code follows.

---

## Motivation

The current grammar is a flat list of `Pattern { symbols: Vec<Symbol>, id: u32 }` where
every symbol is either an atom (interned string ID) or a pattern reference (ID of a
previously learned pattern). This is sufficient for the hierarchical extension (Fix B) but
does not support:

- Non-contiguous patterns (gaps between matched positions)
- IC/OC symbol roles (identification codes vs ordinary content)
- Pattern variables / wildcards
- Stable versioned serialization across these extensions

This doc specifies the grammar model that supports all of the above, and defines the
serialization format that will remain stable as features are added.

---

## Core concepts from SP theory

### Multiple alignment

SP theory's central operation is **multiple alignment**: given a New sequence and a set of
Old patterns, find the alignment that minimises T = G + E where:

- G = cost of the patterns used (grammar cost, paid at training time)
- E = cost of the unmatched New symbols (encoding cost at inference time)
- T = total description length

The alignment is a 2D table. Rows are Old patterns. Columns are New positions. A cell is
filled when a symbol from an Old pattern matches a symbol at that New position. Gaps in a
row mean the pattern matched non-contiguously.

### Symbol roles

In Wolff's SP formulation, every symbol in a pattern has a **role**:

- **OC** (ordinary content) — participates in alignment matching. The symbol is compared
  against New symbols and matched if equal.
- **IC** (identification code) — identifies the pattern without participating in content
  matching. ICs are used to reference a pattern from another pattern (hierarchical
  composition) without "consuming" a New position. At inference time, an IC match means
  "this sub-pattern applies here" rather than "this literal symbol appears here."

Example from Wolff (simplified):

```
Pattern P1: [IC:sentence  OC:the  OC:cat  OC:sat  IC:sentence]
Pattern P2: [IC:sentence  OC:a    OC:dog  OC:ran  IC:sentence]
```

The IC at both ends acts as a bracket — it marks the start and end of the pattern's scope.
A higher-level pattern can reference `IC:sentence` to mean "any sentence pattern applies
here." This is how SP handles hierarchical structure without explicit tree pointers.

### Gaps

A pattern may contain gap markers that match any number of New symbols up to a specified
maximum. Gap markers do not consume a fixed New position — they are placeholders that allow
the pattern to span non-contiguous regions of the New sequence.

```
Pattern P3: [OC:TRIP_EVENT  Gap(0..=5)  OC:RESTORATION]
```

This matches any sequence where TRIP_EVENT appears, followed by RESTORATION within 5
positions, with anything in between.

---

## Grammar model specification

### SymbolRef

Two variants only. No gap, no IC/OC, no variables.

```rust
pub enum SymbolRef {
    Atom(u32),      // interned string ID
    Pattern(u32),   // ID of a learned pattern at a lower level
}
```

`Pattern(pid)` is how hierarchical composition works — a level-1 pattern references
level-0 patterns by ID. No IC/OC mechanism needed; the reference is explicit.

### GapConstraint

Gaps are not symbols — they are **constraints between adjacent symbols** in a pattern.
Stored as a parallel vec alongside `symbols`.

```rust
pub struct GapConstraint {
    pub min: usize,  // minimum New positions to skip between symbol[i] and symbol[i+1]
    pub max: usize,  // maximum New positions to skip (usize::MAX = unbounded)
}
```

`gaps` has length `symbols.len() - 1` or is empty (contiguous pattern, common case).
`gaps[i]` is the constraint between `symbols[i]` and `symbols[i+1]`.

A contiguous pattern has `gaps = []`. A pattern `[A Gap(0,3) B Gap(1,2) C]` has
`symbols = [A, B, C]` and `gaps = [GapConstraint{0,3}, GapConstraint{1,2}]`.

No consecutive-gap invariant needed — the structure makes it impossible.

### Pattern

```rust
pub struct Pattern {
    pub id: u32,
    pub symbols: Vec<SymbolRef>,          // OC content symbols only
    pub gaps: Vec<GapConstraint>,         // empty = contiguous; len = symbols.len()-1 if non-contiguous
    pub frequency: u32,                   // match count during training
    pub level: u8,                        // 0 = atom level
}
```

`frequency` used for calibrated E-score distribution. `level` explicit on pattern for
serialization and lookup (also tracked via `Grammar::levels`).

### GrammarLevel

```rust
pub struct GrammarLevel {
    pub patterns: Vec<Pattern>,   // patterns induced at this level
    pub corpus_e_norms: Vec<f64>, // E_norm of each training sequence at this level (for calibration)
}
```

`levels[0]` = atom level (base patterns). `levels[1]` = first hierarchical level, etc.

### Grammar

```rust
pub struct Grammar {
    pub interner: Interner,            // atom string → u32 ID
    pub levels: Vec<GrammarLevel>,     // levels[0] = atom patterns
    pub e_distribution: EDistribution, // calibration — see calibrated-score-design.md
}
```

`EDistribution` specified in `docs/calibrated-score-design.md`: `sorted_e_norms`,
`threshold`, `level_sorted_e_norms`. Computed in a finalization pass after training.

---

## Serialization format

**Deferred.** Implement once the data model is stable and there are real grammars users
cannot afford to break.

Interim: `serde` derives on all structs, serialize with `serde_json` for development.
When a stable format is needed, evaluate `postcard` (compact, no-schema, pure Rust) or
MessagePack. Do not design magic bytes, CRC32, or migration tables yet.

---

## Inference output (updated InferResult)

Authoritative definition is in `docs/alignment-struct-design.md`. Summary:

```rust
pub struct InferResult {
    pub e_cost: f64,
    pub is_anomaly: bool,
    pub cd: f64,
    pub e_norm: f64,                // E / raw_new_cost, range [0.0, 1.0]
    pub anomaly_percentile: f64,    // rank in training E_norm distribution
    pub level_costs: Vec<f64>,      // raw per-level E costs
    pub level_e_norms: Vec<f64>,    // normalized per-level E costs
    pub alignment: Alignment,       // structured 2D table — see alignment-struct-design.md
}
```

`unmatched` is removed — derive via `alignment.unmatched_symbols() -> Vec<&str>`.
`level_alignments` is removed — level info lives on `AlignmentRow::level`, not a separate vec.

`Alignment`, `AlignmentRow`, `Cell` are fully specified in
`docs/alignment-struct-design.md`. Do not duplicate here.

---

## Implementation order

1. **Grammar data model** (`model.rs`) — `SymbolRef`, `GapConstraint`, `Pattern`, `GrammarLevel`, `Grammar`. No beam change.

2. **Beam → RawAlignment** (`beam.rs`) — beam stays on `u32` IDs, returns `RawAlignment` with match log.

3. **Alignment construction** (`alignment.rs`) — `build_alignment(raw, new_names, grammar)` → `Alignment` with `Display`.

4. **Training loop** (`engine.rs`) — MDL gate, n-gram cold start, N-level outer loop, populates `Grammar`.

5. **Calibrated score** (`engine.rs`) — `infer_internal`, `EDistribution`, E_norm, percentile.

6. **Gap matching** (`beam.rs`) — `can_extend` checks `Pattern::gaps` constraint; no new beam state field.

7. **Serialization** — deferred until model is stable and real users exist.
