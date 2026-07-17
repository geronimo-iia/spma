# Limitations

## Reordering detection requires level-2 pattern formation

SPMA detects reorderings of known structure via the hierarchical grammar — a
sequence `[C, D, A, B]` is anomalous when trained on `[A, B, C, D]` only if
the level-2 grammar learned an ordered pattern `[P_AB, P_CD]`.

Level-2 patterns form only when the corpus is large enough and consistent
enough for the MDL gate to accept them. On small or mixed-order corpora, the
level-2 pass may produce no patterns — leaving reordering undetected while
symbol-level anomalies are still caught correctly.

**Practical consequence**: reordering detection is best-effort. If your corpus
has fewer than ~50 sequences or contains mixed ordering, symbol-level anomalies
(unknown symbols, missing coverage) are reliably detected but reorderings may
not be.

**Workaround**: increase corpus size, or ensure training sequences have
consistent ordering if ordering matters for your use case.

## Public fields allow direct mutation of grammar internals

`Spma::grammar`, `Spma::atom_costs`, and `Grammar` sub-fields are `pub`. Callers can mutate the grammar directly, making invariants (e.g. `atom_costs.len() == interner.len()`) unenforceable. Read access is intentional — inspectability is a core feature. Write access is incidental. Fix in v0.2: read-only accessors (`&Grammar`, `&[f64]`) with `pub(crate)` on fields.

## CLI deps leak into library dependency graph

`anyhow` and `clap` are CLI-only but listed in `[dependencies]`, forcing them on library users. The fix is a Cargo workspace (`spma` lib + `spma-cli` bin). Deferred to v0.2 — not a correctness issue, but annoying for downstream library users.

## F1 ceiling on HDFS without labeled supervision

On the HDFS benchmark, F1=0.893 is the unsupervised ceiling. 92% of false
positives are caused by 5 rare atoms (E6, E16, E18, E25, E28) that also drive
34% of true positives — they cannot be neutralized without anomaly labels.
See `spma-experiments/hdfs-validation/METHOD.md` for the full FP analysis.
