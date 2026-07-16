# Algorithm Assessment — SPMA Train & Infer

Honest evaluation of whether the current algorithm is the right approach, and where SOTA
stands on the same problems.

---

## Is the algorithm correct for what it's trying to do?

Yes, with caveats. The core loop — Shannon costs, MDL gate, beam search, add-only grammar —
is a faithful implementation of Wolff's SP framework for the stated goal: symbolic anomaly
detection on discrete event sequences with an interpretable alignment table as the
explanation. The T=G+E objective is correctly implemented. The beam search with span
contiguity and inter-pattern ordering is the right mechanism for this.

The hierarchical extension (level-N grammar) is also the right direction. Detecting that
`[TRIP_A, BREAKER_OPEN]` followed by `[UNDERVOLTAGE, BACKUP_RELAY]` is anomalous *as an
ordering* — not just as individual symbols — requires exactly this kind of compositional
structure.

---

## Where the algorithm has real weaknesses

### 1. The n-gram cold-start limits what the grammar can learn

The learning loop bootstraps with bigrams and trigrams. This means the grammar can only ever
learn patterns that are contiguous subsequences of the training data. SPMA in Wolff's
original formulation can learn non-contiguous patterns — patterns with gaps — because the
alignment allows skipping. The current implementation discards non-contiguous covered spans
in `extract_learned_patterns`. This is a deliberate simplification, but it means the grammar
cannot represent things like "symbol A always appears somewhere before symbol B, with
anything in between." For log anomaly detection, that is a real limitation.

### 2. The MDL gate is greedy and does not account for pattern interactions

The MDL check adds each candidate pattern if it reduces global T, evaluated greedily in
sorted order. Two patterns that each individually reduce T might together increase it because
they compete for the same coverage. The current implementation does not backtrack. This is a
known limitation of greedy MDL and is acceptable for an exploratory tool, but it means the
grammar is not globally optimal.

### 3. Beam search with fixed K is not guaranteed to find the best alignment

With `keep_rows = 5`, the beam can miss the globally optimal alignment. For short sequences
this is fine. For sequences of 50+ symbols with a large grammar, the beam will increasingly
miss. The performance doc acknowledges this but does not quantify the degradation. There is
no empirical data yet on how often the beam misses on real corpora.

### 4. Costs are recomputed from the joint old+new frequency distribution

Shannon costs are computed from `old_patterns + new_patterns` combined. This means the cost
of a symbol changes as the grammar grows, which is correct for MDL but means T values from
different cycles are not directly comparable. The convergence check (`old_grew ||
added_this_cycle`) sidesteps this correctly, but it also means T cannot be used as an
absolute anomaly threshold across different grammars trained on different corpora.

---

## Has SOTA already solved this?

Depends on what "this" means. There are three distinct problems embedded in SPMA's goals.

### Symbolic sequence anomaly detection

This is well-studied. The main approaches:

**Isolation Forest / One-Class SVM on n-gram features**
Fast, widely deployed, no interpretability, no ordering sensitivity beyond the n-gram window.

**Drain / Spell / LogParse**
Log parsing into templates, then statistical anomaly detection on template sequences. Fast,
widely deployed in production. Drain in particular is the de facto standard for structured
log parsing. No ordering sensitivity beyond simple n-grams. No explanation beyond "this
template sequence was not seen in training."

**DeepLog (Du et al., 2017)**
LSTM trained on normal log sequences. Anomaly = low probability next token. Widely cited,
works well on structured logs. No interpretability, needs retraining on distribution shift,
requires GPU for large corpora.

**LogBERT (Guo et al., 2021)**
BERT on log sequences. Better than DeepLog on some benchmarks. Same interpretability
problem. Higher compute cost.

**What SPMA offers that none of these do**: the alignment table is the explanation. You do
not need SHAP, LIME, or attention visualization — the alignment IS the attribution. For
industrial fault diagnosis where an engineer needs to understand which symbols broke the
pattern, this is a genuine differentiator. No other approach in this space produces a
per-symbol explanation as a native output of the inference algorithm.

**What SPMA does not offer that SOTA does**: probabilistic scoring (SPMA gives a binary E>0
threshold, not a calibrated probability), handling of numeric signals, tolerance for
noisy/partial sequences, and scalability beyond ~1k patterns without the Phase B-F
optimizations described in `docs/performance.md`.

### Grammar induction from sequences

This is where SPMA is most directly comparable to SOTA:

**Sequitur (Nevill-Manning & Witten, 1997)**
Builds a hierarchical context-free grammar from a single sequence by replacing repeated
digrams. Deterministic, O(n) time and space, produces a proper CFG. Much faster than SPMA's
beam-based approach. No anomaly detection built in, but the grammar can be used for it.
Widely used in compression and bioinformatics.

**Re-Pair (Larsson & Moffat, 1999)**
Offline version of Sequitur. Globally optimal pair replacement — better compression ratio
than Sequitur. O(n) with a larger constant. Same limitations: single-sequence, no anomaly
detection, no alignment table.

**ADIOS (Solan et al., 2005)**
Learns hierarchical patterns from a corpus using statistical significance rather than beam
search. Closest in spirit to SPMA — corpus-level, not single-sequence. Produces equivalence
classes of substitutable patterns. No MDL objective, no alignment table.

**The honest comparison**: Sequitur/Re-Pair would give you a hierarchical grammar faster and
with better compression guarantees than SPMA's current beam+MDL approach. But they are
designed for single-sequence compression, not corpus-level anomaly detection, and they do
not produce an alignment table.

### Interpretable anomaly detection specifically

This is the least-solved problem in the space. The closest work:

**Invariant Mining (Lou et al., 2010)**
Mines co-occurrence invariants from log data (e.g. "event A always precedes event B").
Interpretable, fast, but limited to pairwise relationships — no compositional structure.

**LogCluster (Vaarandi & Pihelgas, 2015)**
Clusters log sequences by edit distance, flags outlier clusters. Interpretable at the cluster
level, not at the symbol level.

**Loglizer / LogADL benchmarks**
Systematic comparisons of log anomaly detection methods. SPMA is not in these benchmarks.
The best-performing interpretable method in recent benchmarks is typically Drain + Isolation
Forest, which gives template-level explanations but not symbol-level ones.

No published method produces a per-symbol alignment table as a native inference output.
SPMA's alignment table is genuinely novel in this respect.

---

## Two things worth reconsidering before investing more in the hierarchical extension

### 1. Benchmark against Drain + simple sequence model on a real log dataset

If Drain + LSTM catches 95% of anomalies and SPMA catches 80%, the interpretability argument
needs to be very strong to justify the gap. Without a benchmark on real data, the relative
performance is unknown. The README correctly states "not benchmarked on real data yet" — this
should be the next concrete step before extending the algorithm further.

Suggested datasets:
- HDFS (Xu et al., 2009) — 11M log lines, binary labels, widely used
- BGL (Oliner & Stearns, 2007) — BlueGene/L supercomputer logs, 4.7M lines
- Thunderbird (Oliner & Stearns, 2007) — 211M lines, harder

### 2. Consider Sequitur as the grammar induction backbone

Instead of beam+MDL for training, use Sequitur to build the grammar and then use SPMA's beam
search only at inference time for alignment. Concretely:

- **Train**: concatenate all training sequences with a separator symbol, run Sequitur, extract
  the resulting CFG rules as the Old pattern store.
- **Infer**: run SPMA beam search against the Sequitur grammar, produce the alignment table
  and E cost as today.

This would make training O(n) and deterministic instead of O(n × cycles × beam_k), the
grammar would be globally optimal under the digram substitution criterion, and the alignment
table at inference would be unchanged. The main cost: Sequitur operates on a single
concatenated sequence, so cross-sequence pattern sharing requires careful separator handling.

This is not a small change, but it would resolve the cold-start problem, the greedy MDL
limitation, and the scalability ceiling in one move, while preserving the alignment table
that is SPMA's actual differentiator.

---

## Summary

| Property                 | SPMA current    | Drain+LSTM | Sequitur+SPMA beam |
| ------------------------ | --------------- | ---------- | ------------------ |
| Per-symbol explanation   | ✅ native        | ❌          | ✅ native           |
| Ordering sensitivity     | ✅ level-N       | ❌          | ✅                  |
| Training speed           | ⚠️ O(n×cycles×K) | ✅ fast     | ✅ O(n)             |
| Grammar optimality       | ⚠️ greedy MDL    | n/a        | ✅ globally optimal |
| Probabilistic score      | ❌ binary E>0    | ✅          | ❌ binary E>0       |
| Numeric signals          | ❌               | ✅          | ❌                  |
| Benchmarked on real data | ❌               | ✅          | ❌                  |
| Catastrophic forgetting  | ✅ immune        | ❌          | ✅ immune           |

The algorithm is appropriate for the stated goal. It is not the fastest or most accurate
approach to log anomaly detection, but it is the only approach in this space that produces an
alignment table as a native explanation. That is the actual differentiator and it is worth
preserving. The priority before extending the hierarchical grammar further should be a
benchmark on real data to establish whether the detection rate is competitive enough for the
interpretability advantage to matter.
