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
- Infer (446k sequences, parallel, baseline): ~1816s user, ~123s wall (1480% CPU, 16 cores)
- Infer (446k sequences, parallel, after H+I+J): ~637s user, ~42s wall (~2.85× user, ~2.9× wall)

### Profiling findings (2026-07-18/19, HDFS 446k infer)

`cargo-instruments` Time Profiler, inverted call tree:

| Self time | Function | Root cause |
|-----------|----------|------------|
| ~80s total | `hashbrown::find_inner`, `SipHasher::write`, `hash_one`, `make_hash`, `probe_seq` | `old_cursors`/`new_cursors: HashMap<usize,usize>` in `PartialAlignment` — cloned per candidate per symbol in beam inner loop |
| ~25s | `<f64>::log2` | MDL cost recomputed per candidate per extension — no caching |

After each fix, re-profiled to confirm:

| Profile pass | Top consumer | Action |
|---|---|---|
| Baseline | 40.5% `Map::fold`, 35.9% `hash_one` | HashMap cursors dominate |
| After H (Vec cursors) | 81.2% `Vec::from_iter` | Per-sequence `pid_freq: HashMap` still allocating |
| After pid_freq Vec | 88.3% `LocalKey::with` | TLS overhead exceeded benefit — reverted |
| After grow-only Vec buffers | 83.7% `infer`, 8.5% `log2` | Healthy: hot path owns CPU, no single bottleneck |

### Optimizations applied

**H — Vec cursors in beam.rs**
- Removed dead `old_cursors: HashMap<usize,usize>` (written but never read)
- `new_cursors: HashMap<usize,usize>` → `Vec<u16>` with sentinel `u16::MAX`
- `covered_new: Vec<bool>` → `[u64; 8]` bitmask (512-symbol limit, HDFS max ~298)
- Clone is now memcpy — no heap allocation in beam inner loop

**I — log2 hoisting and pid_freq Vec in engine.rs**
- `fallback_pid = log2(n_prev_pats)` hoisted outside per-sequence loop (level-constant)
- `log2_total = log2(total_pid)` cached once per level per sequence
- `pid_freq: HashMap<u32,u32>` → `Vec<u32>` indexed by pattern id
- Grow-only `pid_freq_buf` / `pid_costs_buf` allocated once per `infer` call, reused across all levels and sequences

**J — TLS attempted, reverted**
Thread-local RefCell buffers caused 88.3% `LocalKey::with` overhead on macOS. Reverted in favour of grow-only Vecs per `infer` call.

## Potential improvements

Accept only if target metric improves ≥2x.

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
Training beam passes are currently sequential per level. Each new pattern is independent within a level — could parallelize with rayon, keeping grammar update sequential.

**E — SIMD symbol comparison** *(after B)*
AVX2: 8 `u32` symbols per instruction.

**F — Arena allocator for beam candidates** *(low priority)*
`bumpalo` arena for short-lived `PartialAlignment` structs. Only relevant at beam_k > 50.

## Order

C ✓ done (no gain). Profile ✓ done. H ✓, I ✓, J attempted+reverted. 2.9× wall speedup achieved.
Remaining: B → D → E → F.
