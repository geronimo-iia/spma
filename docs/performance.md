# Performance

Notes on possible improvements. Only Phase A is implemented. Everything below is unexplored — profile before committing to any of it.

## Targets

| Dataset | Patterns | Symbols each | Target time |
|---|---|---|---|
| Small | 1,000 | 20 | < 5s |
| Medium | 10,000 | 50 | < 10 min |
| Large | 100,000 | 100 | < 4 hours |

## Bottlenecks

1. **Hit detection**: O(new_len × old_pattern_count × max_old_len) — all-against-all, sequential
2. **Beam search**: O(new_len × beam_k × old_pattern_count) per New pattern — dominates at large grammar
3. **Memory**: `Vec<Symbol>` in `Vec<Pattern>` in `Vec<Alignment>` — pointer chasing on every access

## Phases

**A — String interning** ✅
`String::eq` → `u32 ==`. All hot paths operate on u32 IDs.

**B — Structure of Arrays layout**
Replace AoS with flat Vecs + offset table. Expected 3-5x on cost computation and frequency updates.

```rust
pub struct PatternStore {
    ids: Vec<u32>,
    frequencies: Vec<u32>,
    total_costs: Vec<f64>,
    symbol_data: Vec<u32>,      // all symbols concatenated
    symbol_offsets: Vec<u32>,   // symbol_data[offsets[i]..offsets[i+1]] = pattern i
    symbol_costs: Vec<f64>,
    origins: Vec<String>,       // cold, display only
}
```

**C — Inverted index for hit detection**
`symbol_id → Vec<(pattern_id, position)>`. Drops hit detection from O(L × N × avg_len) to O(L × avg_occurrences). Typically 10-100x.

```rust
pub struct SymbolIndex {
    occurrences: Vec<Vec<(u32, u16)>>,  // indexed by symbol_id
}
```

**D — Parallel beam search (rayon)**
Each New pattern is independent — embarrassingly parallel. Grammar update stays sequential per epoch.

```rust
let results: Vec<_> = new_patterns
    .par_iter()
    .map(|new| beam_search(new, &old_store, &symbol_index, beam_k, &costs))
    .collect();
```

**E — SIMD symbol comparison** *(profile first)*
AVX2: 8 `u32` symbols per instruction. `#[cfg(target_feature = "avx2")]` with scalar fallback. Only after B+C.

**F — Arena allocator for beam candidates** *(profile first)*
`bumpalo` arena for short-lived `PartialAlignment` structs. Only relevant when beam_k > 50.

## Order

B → C → D → E → F. Establish `cargo bench` baseline before B. Accept each phase only if target metric improves ≥2x.
