# Add `spma grammar` subcommand

## Goal

Add a `grammar` CLI subcommand that loads a model and prints a structured
summary of the grammar — either human-readable text or machine-readable JSON
suitable for LLM-guided pruning workflows.

This replaces `hdfs-validation/grammar_summary.py` for the LLM pruning
pipeline. The Rust implementation has direct access to all structs (no JSON
re-parsing needed) and produces richer output.

## Why in Rust, not Python

- No JSON round-trip: all data is already in typed structs
- Atom name resolution is exact (interner, not string parsing)
- Per-level e_norm distributions come from `EDistribution` directly
- Single binary, no Python dependency for the prune pipeline

## Files to touch

- `src/bin/spma.rs` — add `Grammar` variant to `Command` enum + handler

No changes to engine or model needed — all required data is already `pub`.

## CLI design

```
spma grammar --model <path> [--json] [--level <N>]
```

- `--json` — emit machine-readable JSON (default: human-readable text)
- `--level N` — restrict output to level N only (default: all levels)

### Clap struct

```rust
/// Print grammar summary (human-readable or JSON)
Grammar {
    /// Path to saved model
    #[arg(short, long)]
    model: String,

    /// Emit JSON output instead of human-readable text
    #[arg(long)]
    json: bool,

    /// Restrict output to a single grammar level (0-indexed)
    #[arg(short, long)]
    level: Option<usize>,
},
```

## Output format

### Human-readable (default)

One block per level:

```
Model: data/model/hdfs_base.json
Beam: 10  MaxGap: 3
Atoms (28): E1 E2 E3 ... E28

Atom costs:
  E1   0.823  ████████  PacketResponder start
  E5   1.102  ██████████████  Receiving block
  ...

Uncovered atoms (always contribute to e_cost): E6 E16 E18 E25 E28

Grammar: 9 levels, 147 patterns

Level 0: 87 patterns, 3 gap patterns, total_freq=347821
  [idx=0  freq= 98432  28.3%] E1(PacketResponder start) → E2(PacketResponder termination)
  [idx=1  freq= 87654  25.2%] E5(Receiving block) → E9(Received block src side)
  ...

Level 1: 12 patterns ...
  [idx=0  freq=  45231  ...] P0 → P1
  ...

E_norm distribution per level (training):
  level      n     p50     p90     p99     max
      0  50000  0.0000  0.1234  0.3210  1.2000
      1  50000  0.1100  0.2500  0.4100  0.9800
      ...
```

**Atom cost annotation:** use a hardcoded map for HDFS events if interner
names match `E\d+` pattern; otherwise print atom name only. The map is the
same as in `grammar_summary.py`:

```rust
fn hdfs_event_desc(name: &str) -> &'static str {
    match name {
        "E1"  => "PacketResponder start",
        "E2"  => "PacketResponder termination",
        "E3"  => "Got exception while serving",
        "E4"  => "Exception in receiveBlock",
        "E5"  => "Receiving block",
        "E6"  => "Received block (dest side)",
        "E7"  => "Served block",
        "E8"  => "Replicating block",
        "E9"  => "Received block (src side)",
        "E10" => "Asking to recover block",
        "E11" => "Received block of size",
        "E12" => "DataXceiver error",
        "E13" => "Deleting block (local)",
        "E14" => "Verification succeeded",
        "E15" => "Exception closing socket",
        "E16" => "Exception writing",
        "E17" => "IOException",
        "E18" => "Receiving empty packet",
        "E19" => "Connection reset",
        "E20" => "BlockReport",
        "E21" => "Deleting block (invalid)",
        "E22" => "allocateBlock",
        "E23" => "addStoredBlock (reportedBlock)",
        "E24" => "addStoredBlock (received)",
        "E25" => "PendingReplicationMonitor timeout",
        "E26" => "addStoredBlock (stored)",
        "E27" => "Unexpected exception",
        "E28" => "DataStreamer exception",
        "E29" => "Recovering block",
        "E30" => "Block is corrupt",
        _     => "",
    }
}
```

If `hdfs_event_desc` returns `""`, omit the description column.

### JSON output (`--json`)

Structured for LLM consumption and programmatic pruning:

```json
{
  "model_path": "data/model/hdfs_base.json",
  "beam_k": 10,
  "max_induced_gap": 3,
  "atoms": [
    {"id": 0, "name": "E1", "cost": 0.823, "description": "PacketResponder start"}
  ],
  "uncovered_atoms": ["E6", "E16", "E18", "E25", "E28"],
  "levels": [
    {
      "level": 0,
      "pattern_count": 87,
      "gap_pattern_count": 3,
      "total_frequency": 347821,
      "patterns": [
        {
          "idx": 0,
          "frequency": 98432,
          "frequency_pct": 28.3,
          "symbols": [
            {"kind": "atom", "id": 0, "name": "E1", "description": "PacketResponder start"},
            {"kind": "atom", "id": 1, "name": "E2", "description": "PacketResponder termination"}
          ],
          "gaps": [],
          "rendered": "E1 → E2"
        }
      ],
      "e_norm": {"p50": 0.0, "p90": 0.1234, "p99": 0.3210, "max": 1.2000, "n": 50000}
    }
  ],
  "threshold": 0.2,
  "atom_costs": [0.823, 1.102, ...]
}
```

For gap patterns, `gaps` is an array of `{"min": N, "max": M}` objects,
one per adjacent symbol pair (length = symbols.len() - 1).

For level > 0, symbols with `"kind": "pattern"` have `"id"` pointing to
the pattern index in the previous level's `patterns` array.

The `rendered` field is a human-readable string:
- Atom: `"E1"` (or `"E1(PacketResponder start)"` if description available)
- Pattern ref: `"P0"` (P + index)
- Gap between symbols i and i+1: `"~[min,max]→"` if gap present, `"→"` if contiguous

## Implementation notes

Access model data through `Spma` public fields:
```rust
let spma = Spma::load(BufReader::new(f))?;
let names  = &spma.grammar.interner.names;     // Vec<String>
let levels = &spma.grammar.levels;             // Vec<GrammarLevel>
let dist   = &spma.grammar.e_distribution;
let costs  = &spma.atom_costs;                 // Vec<f64>
```

Check whether `Interner` exposes `names` as a public field or via method —
adapt to actual API.

Uncovered atoms: atoms not referenced by any `SymbolRef::Atom(_)` across all
patterns in all levels.

For e_norm distribution per level: use `dist.level_sorted_e_norms[i]` if
`i < dist.level_sorted_e_norms.len()`, else skip.

Output to stdout. Use `BufWriter` for performance.

## Tests

```rust
#[test]
fn grammar_json_output_round_trips() {
    // Train a tiny model (5 seqs, 3-token vocab)
    // Run grammar --json, parse the JSON output
    // Assert: levels.len() matches spma.grammar.levels.len()
    // Assert: atoms array length matches interner
    // Assert: all pattern idx values are contiguous 0..N
}
```

## Verification

```bash
cargo build --release

# Human-readable
./target/release/spma grammar --model hdfs-validation/data/model/hdfs_base.json

# JSON for LLM pruning pipeline
./target/release/spma grammar --model hdfs-validation/data/model/hdfs_base.json --json \
  | python -m json.tool > /tmp/grammar.json

# Confirm JSON is valid and contains expected keys
python -c "
import json
g = json.load(open('/tmp/grammar.json'))
assert 'levels' in g and 'atoms' in g and 'uncovered_atoms' in g
print('levels:', len(g['levels']))
print('patterns level0:', len(g['levels'][0]['patterns']))
print('uncovered:', g['uncovered_atoms'])
"
```

## Decision rule

Keep if:
- `cargo test` passes
- JSON output is valid and parseable
- Human-readable output matches current `grammar_summary.py` output for same model
