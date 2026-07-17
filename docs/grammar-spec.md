# Grammar Model

Reference for the core data model. Code in `src/model.rs` is authoritative for field names and types; this doc explains the design rationale and what was deliberately excluded.

## SymbolRef

```rust
pub enum SymbolRef {
    Atom(u32),      // interned string ID
    Pattern(u32),   // ID of a learned pattern at a lower level
}
```

`Pattern(pid)` is how hierarchical composition works. A level-1 pattern holds references to level-0 patterns. No tree pointers needed — the ID is the reference.

## GapConstraint

Gaps are constraints between adjacent symbols, not symbols themselves. Stored as a parallel vec alongside `symbols`:

```rust
pub struct GapConstraint {
    pub min: usize,  // minimum New positions to skip between symbol[i] and symbol[i+1]
    pub max: usize,  // maximum New positions to skip (usize::MAX = unbounded)
}
```

`gaps` has length `symbols.len() - 1` or is empty (contiguous pattern, common case).

A contiguous pattern: `gaps = []`.  
Pattern `[A Gap(0,3) B]`: `symbols=[A,B]`, `gaps=[GapConstraint{0,3}]`.

## Pattern

```rust
pub struct Pattern {
    pub id: u32,
    pub symbols: Vec<SymbolRef>,
    pub gaps: Vec<GapConstraint>,    // empty = contiguous
    pub frequency: u32,
    pub level: u8,
}
```

## Grammar

```rust
pub struct Grammar {
    pub interner: Interner,
    pub levels: Vec<GrammarLevel>,
    pub e_distribution: EDistribution,
}

pub struct GrammarLevel {
    pub patterns: Vec<Pattern>,
}
```

`levels[0]` = atom-level patterns. `levels[1]` = first hierarchical level, etc.

`EDistribution` — see [docs/scoring.md](scoring.md).

## What was NOT implemented and why

### IC/OC symbol roles

In Wolff's full SP formulation, every symbol carries a role: OC (ordinary content, participates in alignment) or IC (identification code, used as a bracket/reference without consuming a New position). IC symbols allow a higher-level pattern to assert "any sentence applies here" without explicitly listing sentence content.

**Not implemented.** `SymbolRef::Pattern(pid)` handles hierarchical composition explicitly: a level-1 pattern holds IDs of level-0 patterns. This is unambiguous and does not require IC semantics. IC/OC would add complexity without changing the MDL objective or the induction algorithm. If full SP-theory compatibility is needed later, IC can be added as a third `SymbolRef` variant without breaking existing serialized grammars.

### Pattern variables / wildcards

Not in scope. Gap patterns (`GapConstraint`) cover the common case (bounded skip). True wildcards (match any symbol) would require a different beam state and a separate grammar cost model.

### Serialization stability

Current: `serde` derives, serialize with `serde_json`. Acceptable for research and model files. If production deployment requires forward-compatible binary format, evaluate `postcard` or MessagePack at that point. Do not add magic bytes, CRC32, or migration tables until there are real users who cannot afford to break saved grammars.
