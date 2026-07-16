# Examples

Run any example with:

```bash
cargo run --example fault_detection
cargo run --example ordered_sequences
```

## Examples

| File | What it shows |
|---|---|
| `fault_detection.rs` | Varied corpus, gap patterns, anomaly scoring |
| `ordered_sequences.rs` | Order sensitivity, grammar coverage limits |

## Reading the output

Each inferred sequence prints a score line followed by an alignment table.

### Score line

```
[ANOMALY]  E=3.415  CD=+7.000  — label
```

| Field | Meaning |
|---|---|
| `E` | Error cost (bits). Bits spent on uncovered symbols. `E=0` → perfect grammar coverage. |
| `CD` | Compression delta (bits). Bits saved by firing grammar patterns. `CD=0` → no patterns matched. |
| `T` | Total cost = `E + CD`. Shown in the footer. |
| `is_anomaly` | `true` when `e_norm = E / raw_cost > threshold` (default threshold = 0.0). |

E and CD trade off: a known sequence maximises CD and minimises E. A completely novel sequence has `E = full raw cost`, `CD = 0`.

Anomaly gating uses only `e_norm`, not `CD`.

### Alignment table

```
        TRIP_A        BREAKER_OPEN  OVERCURRENT   BACKUP_RELAY
P3(L0)  TRIP_A        .             .             .
P0(L0)  .             BREAKER_OPEN  <1>           BACKUP_RELAY
---
E: 3.4 bits   CD: 7.0 bits   T: 3.4 bits
```

- Each row is one learned pattern. `P0(L0)` = pattern id 0, grammar level 0.
- `.` = position not covered by this pattern.
- `<1>` = gap cell. The pattern has a gap constraint here; 1 symbol was skipped. The skipped symbol is **uncovered** and contributes to E.
- Footer: `E`, `CD`, `T` rounded to 1 decimal place.

### Atom cost

With frequency-based costs, each atom costs `-log2(freq / total)`. Rare atoms cost more than frequent ones. An atom that never enters any learned pattern stays uncovered at inference — it contributes its full cost to E even on training sequences. This is expected: **grammar coverage depends on corpus size and frequency thresholds**, not just on an atom being present.

## Known limits

- **Grammar coverage**: atoms that never co-occur frequently enough to form a pattern remain uncovered forever. Training sequences can therefore score `E > 0` and `is_anomaly = true` if the corpus is too small or too varied.
- **Order detection**: reliable only with a homogeneous corpus. A varied corpus learns shorter patterns that can stitch across reorderings, reducing the anomaly signal for out-of-order sequences.
- **Missing symbols**: not detected unless the removed atom was covered by a pattern (its absence leaves other pattern symbols unmatched).
