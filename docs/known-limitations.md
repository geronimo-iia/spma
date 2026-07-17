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

## F1 ceiling on HDFS without labeled supervision

On the HDFS benchmark, F1=0.893 is the unsupervised ceiling. 92% of false
positives are caused by 5 rare atoms (E6, E16, E18, E25, E28) that also drive
34% of true positives — they cannot be neutralized without anomaly labels.
See [spma-experiments/hdfs-validation/METHOD.md](https://github.com/geronimo-iia/spma-experiments/blob/main/hdfs-validation/METHOD.md) for the full FP analysis.
