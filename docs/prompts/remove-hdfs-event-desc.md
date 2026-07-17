# Remove `hdfs_event_desc` from spma binary

## Goal

Remove the hardcoded HDFS event description map from `src/bin/spma.rs`.
It is dataset-specific and does not belong in a general-purpose library binary.

After this change:
- Human-readable output (`spma grammar`) shows atom names only (e.g. `E5`)
- JSON output (`spma grammar --json`) has `"name": "E5"` with no `"description"` field
- Callers that need descriptions supply their own lookup (Python script, LLM prompt, etc.)

## File to touch

- `src/bin/spma.rs` only

## Changes

### 1. Delete `hdfs_event_desc` function

Remove lines defining `fn hdfs_event_desc(name: &str) -> &'static str { ... }` entirely.

### 2. Update `render_symbol`

Current:
```rust
SymbolRef::Atom(id) => {
    let name = interner.name(*id);
    let desc = hdfs_event_desc(name);
    if desc.is_empty() {
        name.to_owned()
    } else {
        format!("{}({})", name, desc)
    }
}
```

Replace with:
```rust
SymbolRef::Atom(id) => interner.name(*id).to_owned(),
```

### 3. Update JSON output

Find every place `hdfs_event_desc` is called when building the JSON `atoms` array
or pattern symbol objects. Remove the `"description"` field from those JSON
structs/serializations entirely.

For example, if the JSON atom object is:
```json
{"id": 0, "name": "E5", "description": "Receiving block", "cost": 2.861}
```
it becomes:
```json
{"id": 0, "name": "E5", "cost": 2.861}
```

Same for pattern symbol objects — remove any `"description"` key.

### 4. Verify all call sites removed

After changes, `grep -n hdfs_event_desc src/bin/spma.rs` must return no matches.

## Tests

No new tests needed. Existing `cargo test` must pass.

Manually verify:
```bash
cargo build --release

# Human-readable: atom names only, no parenthetical descriptions
./target/release/spma grammar \
    --model hdfs-validation/data/model/hdfs_base.json \
    | grep "E5"
# Expected: "E5" with cost bar, no "(Receiving block)"

# JSON: no "description" field
./target/release/spma grammar \
    --model hdfs-validation/data/model/hdfs_base.json \
    --json | python3 -c "
import json, sys
g = json.load(sys.stdin)
for a in g['atoms']:
    assert 'description' not in a, f'description still present: {a}'
print('ok — no description fields')
"
```

## Decision rule

Keep if `cargo test` passes and grep confirms no `hdfs_event_desc` remaining.
