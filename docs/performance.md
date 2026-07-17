# Performance

## Current implementation

- **String interning**: all hot paths operate on `u32` IDs (`symbol_id`, `pattern_id`). No `String::eq` in beam search or cost computation.
- **Parallel infer**: `spma infer` parallelizes sequence scoring with rayon. Sequences are independent — embarrassingly parallel. Stdout buffered and flushed after all workers complete.
- **mimalloc**: global allocator replaced with mimalloc for parallel allocation throughput.
- **MDL cache**: total E cost cached during training beam passes — avoids a full extra pass to build `e_distribution`.

Observed on HDFS (1k training corpus, 446k infer):
- Training (1k sequences): seconds
- Infer (446k sequences, parallel): minutes

## Potential improvements

Profile before committing to any of these. Accept only if target metric improves ≥2x.

**B — Structure of Arrays layout**
Replace AoS with flat Vecs + offset table. Expected 3–5x on cost computation and frequency updates.

```rust
pub struct PatternStore {
    symbol_data: Vec<u32>,      // all symbols concatenated
    symbol_offsets: Vec<u32>,   // symbol_data[offsets[i]..offsets[i+1]] = pattern i
    frequencies: Vec<u32>,
    total_costs: Vec<f64>,
}
```

**C — Inverted index for hit detection**
`symbol_id → Vec<(pattern_id, position)>`. Drops hit detection from O(L × N × avg_len) to O(L × avg_occurrences). Expected 10–100x on large grammars.

```rust
pub struct SymbolIndex {
    occurrences: Vec<Vec<(u32, u16)>>,  // indexed by symbol_id
}
```

**D — Parallel training**
Training beam passes are currently sequential per level. Each New pattern is independent within a level — could parallelize with rayon, keeping grammar update sequential.

**E — SIMD symbol comparison** *(profile first)*
AVX2: 8 `u32` symbols per instruction. Only after B+C are done.

**F — Arena allocator for beam candidates** *(profile first)*
`bumpalo` arena for short-lived `PartialAlignment` structs. Only relevant at beam_k > 50.

**G — Redundant level-0 beam search in infer** *(done)*
`best_raw.match_log` now seeded into `prev_match_log` before the N-level loop. Level=1 extracts pid_seq directly from it; no second beam call. Each subsequent level passes its beam result's match_log forward.

## Order

G ✓ done. Remaining: B → C → D → E → F. Establish `cargo bench` baseline before B.
