# Performance

## Current implementation

- **String interning**: all hot paths operate on `u32` IDs (`symbol_id`, `pattern_id`). No `String::eq` in beam search or cost computation.
- **Parallel infer**: `spma infer` parallelizes sequence scoring with rayon. Sequences are independent — embarrassingly parallel. Stdout buffered and flushed after all workers complete.
- **mimalloc**: global allocator replaced with mimalloc for parallel allocation throughput.
- **MDL cache**: total E cost cached during training beam passes — avoids a full extra pass to build `e_distribution`.
- **Infer match_log reuse**: level-0 `beam_search` result's `match_log` seeded directly into the N-level loop; level=1 extracts pid_seq from it without a second beam call.
- **SymbolIndex on GrammarLevel**: inverted index `symbol_id → [(pattern_idx, pos)]` built once per level, eliminates per-call `HashMap` rebuild in `beam_search`. Measured gain on HDFS 50k/446k: <1% — bottleneck is beam candidate expansion, not index construction.

Observed on HDFS (50k training corpus, 446k infer, release build, Apple Silicon):
- Training (50k sequences): ~138s user, ~1m48s wall (sequential, 128% CPU)
- Infer (446k sequences, parallel): ~1816s user, ~2m3s wall (1480% CPU, 16 cores)

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

**D — Parallel training**
Training beam passes are currently sequential per level. Each New pattern is independent within a level — could parallelize with rayon, keeping grammar update sequential.

**E — SIMD symbol comparison** *(profile first)*
AVX2: 8 `u32` symbols per instruction. Only after B+C are done.

**F — Arena allocator for beam candidates** *(profile first)*
`bumpalo` arena for short-lived `PartialAlignment` structs. Only relevant at beam_k > 50.

## Order

C ✓ done (no gain). **Profile before B** — use `cargo flamegraph` or `samply` to confirm
beam candidate expansion is the bottleneck before investing in SoA layout.
Remaining: profile → B → D → E → F.
