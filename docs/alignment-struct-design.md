# Alignment Struct Design — Phase 1b

Specification for the structured 2D alignment table that replaces the current
`alignment: String` field in `InferResult`. This is a clean-slate design — no
compatibility constraint with the current string format.

Depends on: `docs/grammar-spec.md` (grammar model, symbol roles, SymbolRef variants).

---

## What the alignment table is

The alignment table is the central output of SPMA inference. It is a 2D structure:

- **Columns** = positions in the New sequence (left to right)
- **Rows** = Old patterns that participated in the best alignment

A cell `(row, col)` is filled when symbol at Old pattern position `p` matched New position
`col`. A cell is empty when the Old pattern "skipped" that New position.

Example — New = `[A B C D E F]`, grammar has `P1=[A B C]`, `P2=[D E F]`, level-2 has
`P3=[P1 P2]`:

```
         A    B    C    D    E    F
P1       A    B    C
P2                      D    E    F
P3(L2)  [P1]            [P2]
```

Row `P3` is a level-2 match: its symbols are pattern references, not atoms. The cell
contains the referenced pattern's label, not a literal symbol. Columns covered by P1 are
"claimed" by both P1 (content match) and P3 (structural match).

This table is the explanation. An engineer reading it sees exactly which grammar patterns
covered which events, at which level of abstraction.

---

## Data model

### Cell

A cell represents one match between an Old pattern symbol and a New position.

```rust
pub struct Cell {
    pub old_pos: usize,       // position within the Old pattern (0-indexed)
    pub new_pos: usize,       // position within the New sequence (0-indexed)
    pub content: String,      // resolved display string (atom name or "P{id}")
    pub is_gap: bool,         // true if this cell represents a gap span
    pub gap_span: usize,      // number of New positions spanned (0 if not a gap)
    pub cost: f64,            // bit cost (0.0 for gaps)
}
```

No `CellContent` enum — `String` + two gap fields cover all cases without pattern
matching overhead on the hot display path. Gap cells have `is_gap=true`,
`content = format!("<{}>", gap_span)`, `gap_span = new_end - new_start`.

### AlignmentRow

One row = one Old pattern participating in the alignment.

```rust
pub struct AlignmentRow {
    pub pattern_id: u32,
    pub pattern_label: String,   // human-readable name, e.g. "P42" or interner string
    pub level: usize,            // grammar level where this pattern lives (0 = atom level)
    pub cells: Vec<Cell>,        // only filled cells — sparse representation
    pub fully_matched: bool,     // true if every symbol in the Old pattern was matched
}
```

Sparse: `cells` contains only matched positions, not a full `new_len`-wide array. Callers
that need a dense grid build it from `cell.new_pos` indexing. This keeps the struct cheap
to construct and avoids allocating a wide empty grid for patterns that only match a few
positions.

**Ordering invariant**: `cells` is guaranteed sorted by `new_pos` ascending. Callers may
rely on this without re-sorting. Guaranteed at construction in `finalize()`.

```rust
impl AlignmentRow {
    /// First New position matched by this row. None if row has no cells (should not occur).
    pub fn first_new_pos(&self) -> Option<usize> {
        self.cells.first().map(|c| c.new_pos)
    }
}
```

### Alignment

The complete alignment result for one infer call.

```rust
pub struct Alignment {
    /// Symbol names of the New sequence, in order.
    pub new_symbols: Vec<String>,

    /// One row per Old pattern that participated (at least one cell filled).
    ///
    /// **Ordering invariant**: sorted by `(level, first_new_pos)` ascending — atom-level
    /// patterns first, then higher levels; within a level, left-to-right in New.
    /// Callers may rely on this without re-sorting. Guaranteed at construction in `finalize()`.
    pub rows: Vec<AlignmentRow>,

    /// covered[i] = true if New position i was matched by at least one Old pattern.
    pub covered: Vec<bool>,

    /// Total E cost: sum of bit_cost for all uncovered New positions.
    pub e_cost: f64,

    /// Per-level E costs. level_costs[0] = level-1 alignment cost, etc.
    /// Empty if grammar has only one level.
    pub level_costs: Vec<f64>,
}
```

`Alignment` is the only type the caller needs. `Cell`, `CellContent`, `AlignmentRow` are
pub so callers can pattern-match on them, but the primary surface is `Alignment`.

---

## Construction

`Alignment` is constructed inside the inference engine, not by the caller. The beam search
already tracks the data needed — the change is preserving it through `finalize()` instead
of discarding it.

### What the beam currently tracks (sufficient for construction)

In `PartialAlignment`:
- `old_cursors: HashMap<usize, usize>` — last matched `old_pos` per `old_idx`
- `new_cursors: HashMap<usize, usize>` — last matched `new_pos` per `old_idx`
- `covered_new: Vec<bool>`

This is the **final state** only — the last matched position per pattern. It is not enough
to reconstruct all matched positions (for a 3-symbol pattern matched at new[2,3,4], we
only know new[4] from `new_cursors`).

### Ordering guarantees — construction contracts

`finalize()` is solely responsible for establishing both invariants. No caller sorts.

| Invariant | Established by | Method |
|---|---|---|
| `AlignmentRow::cells` sorted by `new_pos` asc | `finalize()` | sort `match_log` by `(old_idx, new_pos)` before grouping |
| `Alignment::rows` sorted by `(level, first_new_pos)` asc | `finalize()` | sort rows after construction |

Both sorts use `sort_unstable_by_key` — O(n log n), no allocations, stable enough because
the keys are total (ties broken by the second key or by `usize::MAX` sentinel).

### Required beam change: match log

To construct the full alignment, the beam records every match event. `PartialAlignment`
does **not** own a `Vec<MatchEvent>` — cloning it on every beam fork would be O(N×K×M).
Instead, `beam_search` owns a `MatchArena`; each `PartialAlignment` owns only a single
`u32` index (the tail of a linked list).

```rust
pub struct MatchEvent {
    pub old_idx: usize,
    pub old_pos: usize,
    pub new_pos: usize,
    pub cost: f64,
}

struct MatchNode {
    event: MatchEvent,
    parent: Option<u32>,  // index into MatchArena::nodes
}

struct MatchArena {
    nodes: Vec<MatchNode>,
}

impl MatchArena {
    fn push(&mut self, event: MatchEvent, parent: Option<u32>) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(MatchNode { event, parent });
        idx
    }

    fn collect(&self, tail: Option<u32>) -> Vec<MatchEvent> {
        let mut out = Vec::new();
        let mut cur = tail;
        while let Some(i) = cur {
            let node = &self.nodes[i as usize];
            out.push(node.event.clone());
            cur = node.parent;
        }
        out.reverse();
        out
    }
}

// PartialAlignment gains:
log_tail: Option<u32>,  // replaces match_log: Vec<MatchEvent>
```

Forking a `PartialAlignment` copies one `u32` (the tail index), not a growing vec.
`extend_match` calls `arena.push(event, self.log_tail)` and stores the returned index.
`extend_skip` copies `log_tail` unchanged — zero allocation.

`finalize()` calls `arena.collect(winning.log_tail)` once, for the single winning
alignment only. O(M) walk of the linked list, no allocations during beam search.

`MatchArena` is owned by the `beam_search` stack frame, passed `&mut` to every
`extend_match` call.

### build_alignment produces Alignment

```rust
pub fn build_alignment(raw: &RawAlignment, new_names: &[&str], grammar: &Grammar) -> Alignment {
    // 1. Sort match_log by (old_idx, new_pos).
    let mut log = raw.match_log.clone();
    log.sort_unstable_by_key(|e| (e.old_idx, e.new_pos));

    // 2. Group by old_idx → one AlignmentRow per old_idx.
    //    For each MatchEvent, resolve content:
    //      pattern.symbols[old_pos]: SymbolRef::Atom(id) → new_names[new_pos]
    //                                SymbolRef::Pattern(id) → format!("P{id}")
    //      If pattern.gaps is non-empty and old_pos < gaps.len():
    //        insert a gap Cell after this cell: is_gap=true, gap_span computed from
    //        next_match.new_pos - this.new_pos - 1

    // 3. Sort rows by (level, first_new_pos) ascending.
    rows.sort_unstable_by_key(|r| (r.level, r.first_new_pos().unwrap_or(usize::MAX)));

    // 4. e_cost already in raw.e_cost. Compute e_norm = e_cost / raw_new_cost.

    // 5. Return Alignment { new_symbols, rows, covered, e_cost, e_norm, level_costs, level_e_norms }.
}
```

---

## Display

`Display` on `Alignment` renders the classic SP alignment table format. This is the
human-readable form for logs and debug output.

### Format

```
         A    B    C    D    E    F
P1(L0)   A    B    C    .    .    .
P2(L0)   .    .    .    D    E    F
P3(L2)  [P1]  .    .   [P2]  .    .
-----------------------------------------
E:       0.0 bits   CD: 12.3 bits   T: 12.3 bits
```

Rules:
- Column width = max symbol name length + 2, minimum 4
- Covered positions: `cell.content` (atom name or `P{id}` for pattern refs)
- Uncovered positions: `.`
- Gap cells: `cell.content` already contains `<N>` (set during `build_alignment`)
- Row label: `{pattern_label}(L{level})`, left-aligned, fixed width
- Footer line: E, CD, T costs in bits

### Implementation

```rust
impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 1. Compute column width
        // 2. Write header row (New symbol names)
        // 3. For each row in self.rows:
        //    - build dense Vec<Option<&Cell>> of length new_symbols.len()
        //    - write row label, then each column
        // 4. Write separator + footer
    }
}
```

No external dependency. Pure `fmt::Write` calls.

---

## InferResult (updated)

Authoritative definition. `src/lib.rs`:

```rust
pub struct InferResult {
    pub e_cost: f64,
    pub is_anomaly: bool,
    pub cd: f64,
    pub e_norm: f64,                // E / raw_new_cost, range [0.0, 1.0]
    pub anomaly_percentile: f64,    // rank in training E_norm distribution
    pub level_costs: Vec<f64>,      // raw per-level E costs
    pub level_e_norms: Vec<f64>,    // normalized per-level E costs
    pub alignment: Alignment,       // structured 2D table
}
```

Removed: `alignment: String`, `level_alignments: Vec<String>`, `unmatched: Vec<String>`.
Level info lives on `AlignmentRow::level`. Unmatched symbols derived via:

```rust
impl Alignment {
    /// Symbol names of New positions not covered by any Old pattern.
    /// Returned in New-position order.
    pub fn unmatched_symbols(&self) -> Vec<&str> {
        self.covered.iter().enumerate()
            .filter(|(_, &c)| !c)
            .map(|(i, _)| self.new_symbols[i].as_str())
            .collect()
    }
}
```

No `unmatched` field on `InferResult` — derive on demand from `alignment.unmatched_symbols()`.

`Display` on `InferResult` delegates to `alignment.fmt()`.

---

## beam_search signature

Beam stays on `u32` IDs throughout. Name resolution happens once, outside the beam, when
constructing `Alignment` from `RawAlignment`.

```rust
// beam.rs — unchanged ID-based interface
pub fn beam_search(
    new: &[u32],
    old: &[&Pattern],   // needs Pattern for gap constraints; symbols stay u32
    beam_k: usize,
    costs: &[f64],
) -> Vec<RawAlignment>

pub struct RawAlignment {
    pub match_log: Vec<MatchEvent>,  // every (old_idx, old_pos, new_pos, cost) event
    pub covered: Vec<bool>,
    pub e_cost: f64,
    pub cd: f64,
}

pub struct MatchEvent {
    pub old_idx: usize,
    pub old_pos: usize,
    pub new_pos: usize,
    pub cost: f64,
}

// alignment.rs — separate step
pub fn build_alignment(
    raw: &RawAlignment,
    new_names: &[&str],
    grammar: &Grammar,
) -> Alignment
```

`build_alignment` does all name resolution and gap annotation. Beam is testable in
isolation with no string or Grammar dependency.

---

## What this unblocks

- **Gap matching** (Phase 2a): `Cell::is_gap` + `gap_span` already defined. `build_alignment`
  inserts gap cells when `Pattern::gaps` is non-empty. No struct change when gap matching
  is implemented.
- **Calibrated score** (Phase 1c): `RawAlignment::e_cost` feeds directly into percentile
  lookup. `build_alignment` propagates it to `Alignment::e_cost` and computes `e_norm`.
- **Programmatic consumers**: iterate `alignment.rows`, check `row.fully_matched`, filter
  by `row.level`, call `alignment.unmatched_symbols()` — no string parsing.

---

## What is NOT in scope for 1b

- Gap matching logic — `Cell::is_gap` is defined but `can_extend` gap constraint check is Phase 2a
- Calibrated E-score — `e_norm` field exists but percentile table is Phase 1c
- IC/OC — not in scope at all; `SymbolRef::Pattern(pid)` handles hierarchical composition
- Multi-level alignment — `level_costs` kept, level-N rows merged into `Alignment::rows` with `level` field
