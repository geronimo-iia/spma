# Algorithm Assessment 2 — Full SP Theory Implementation Roadmap

Follow-up to `algorithm-assessment.md`. That doc assessed the current implementation
against SOTA. This doc answers the next question: if we implement the full SP theory
feature set, what does that look like, in what order, and what does each step buy us?

---

## What "full SP theory" actually means

Wolff's SP framework has five distinguishing properties. The current SPMA implementation
has two of them. The table:

| Property | Current SPMA | Full SP |
|---|---|---|
| T=G+E MDL objective | ✅ | ✅ |
| Hierarchical grammar (N levels) | ✅ (Fix B) | ✅ |
| Non-contiguous patterns (gaps) | ❌ | ✅ |
| IC/OC symbol roles | ❌ | ✅ |
| Structured 2D alignment table | ❌ (string only) | ✅ |
| Calibrated anomaly score | ❌ (binary E>0) | ✅ |
| Pattern variables / unification | ❌ | reserved |

The three missing core features are non-contiguous patterns, IC/OC roles, and the
structured alignment table. Each has a concrete implementation path. Variables are
reserved in the schema (see `grammar-spec.md`) but not scheduled — they require a
unification engine and the use cases in the EDF fault domain don't clearly need them yet.

---

## Why grammar spec first

Before implementing any of the above, the grammar data model and serialization format must
be locked. Reason: adding `SymbolRole` after the fact requires migrating every `Symbol` in
every serialized file. Adding `GapSpec` after the fact requires a new `SymbolRef` variant
that existing serialized patterns don't encode. If we add features before the schema is
stable, every feature adds a migration tax.

The grammar spec (`docs/grammar-spec.md`) defines the v2 format that supports all of the
above. Implementation starts there, not with the features themselves.

---

## Feature assessments

### Feature A: Structured 2D alignment table

**What it adds**: the alignment table is SPMA's actual differentiator. Currently it is
serialized to a debug string and thrown away. A structured `Alignment` struct lets callers
do things the string cannot: count unmatched symbols programmatically, extract which
patterns covered which positions, build a UI that highlights anomalous positions, integrate
with domain knowledge (this unmatched position is sensor X, owned by substation Y).

**Implementation cost**: low. The beam already tracks `old_cursors` and `new_cursors`
internally — the data exists. The change is surfacing it instead of discarding it. No
algorithmic change.

**Dependency**: none. Can be done before or after grammar spec changes.

**ROI**: high. This is the feature an engineer at EDF actually uses. Everything else is
infrastructure. Ship this before adding new algorithmic complexity.

---

### Feature B: Calibrated anomaly score

**What it adds**: replaces binary `is_anomaly: bool` (E > threshold) with
`anomaly_percentile: f64` (where does this E value fall in the training distribution?).

Example: a sequence with E=4.2 bits is more informative than "anomalous" — it means "this
sequence costs more to encode than 97% of training sequences." An operator can set their
own threshold. A downstream ML pipeline can use the percentile as a feature.

**Implementation cost**: very low. Training already computes E on the corpus (implicitly,
via MDL gate). Store the sorted distribution, emit a percentile at inference via binary
search. No beam change, no grammar change.

**Dependency**: none. Independent of grammar spec, independent of other features.

**ROI**: high. Turns a binary classifier into something usable by downstream systems.
Almost free to implement.

---

### Feature C: Non-contiguous patterns (gap matching)

**What it adds**: patterns with gaps — `[TRIP_EVENT  Gap(0..=5)  RESTORATION]` matches
TRIP_EVENT followed by RESTORATION within 5 positions with anything in between. This is
the most common structural relationship in industrial fault logs: "event A eventually
followed by event B, within a time window."

Without this, SPMA can only detect anomalies in the exact sequence of directly adjacent
symbols. Two events that always co-occur but with variable spacing between them cannot be
learned as a pattern today. This is a real limitation for the EDF use case.

**Implementation cost**: medium.

- `SymbolRef::Gap(GapSpec)` variant in the grammar model (already in spec)
- Beam `can_extend`: when next Old symbol is a Gap, allow `new_pos` in range
  `[prev_new + gap.min, prev_new + gap.max]`
- `extract_learned_patterns`: when building candidate patterns from beam alignments,
  detect non-contiguous covered spans and insert Gap markers
- N-gram miner: add a gap-aware mining pass for the cold-start phase

The gap-aware miner is the hardest part. A naive implementation has O(n²) pairs to check
per sequence. With a max_gap cap (e.g. 10) it stays O(n × max_gap) which is fine.

**Dependency**: grammar spec (needs `SymbolRef::Gap` in the schema before patterns with
gaps can be serialized).

**ROI**: high for fault log use case. Essential if training sequences have variable-length
inter-event intervals.

---

### Feature D: IC/OC symbol roles

**What it adds**: IC (identification code) symbols act as pattern brackets — they label a
pattern's identity without consuming New positions in the alignment. This enables:

1. **Pattern disambiguation**: two patterns with the same content but different ICs are
   distinct. Alignment can distinguish "this is a TRIP_SEQUENCE" from "this is a
   STARTUP_SEQUENCE" even if they share sub-patterns.

2. **Hierarchical reference without consuming positions**: a level-2 pattern can reference
   a level-1 pattern via IC match rather than symbol-by-symbol match, enabling true
   recursive composition.

3. **Structural analogy**: Wolff uses IC/OC to model syntax, semantics, and world
   knowledge in a unified representation. For fault logs, this maps to: IC labels the
   event class (UNDERVOLTAGE, OVERCURRENT), OC symbols are the specific event IDs.

**Implementation cost**: medium-high.

- `SymbolRole` field on `Symbol` (low cost, already in spec)
- Beam search: IC match does not advance New cursor, checks pattern-identity context
  instead of symbol equality. Requires tracking "active pattern context" in
  `PartialAlignment` state — currently not tracked.
- Grammar induction: IC symbols must be inserted during learning, which requires the
  engine to decide when to assign IC vs OC role. Heuristic needed: symbols that appear
  exclusively at pattern boundaries (first/last position) are candidates for IC role.

The beam change is the hard part. IC matching introduces a new kind of state into the
partial alignment that doesn't fit cleanly into the current `old_cursors`/`new_cursors`
model. Expect 3-4 days of careful work and test coverage.

**Dependency**: grammar spec, structured alignment table (IC matches need to appear in the
alignment table as pattern-identity annotations, not symbol matches).

**ROI**: medium for immediate EDF use case, high for long-term expressiveness. The fault
log use case can be largely served by gaps + hierarchical grammar without IC/OC. But IC/OC
is what makes SPMA a general symbolic reasoning system rather than a specialized anomaly
detector. Worth building after benchmarking confirms the system is detection-competitive.

---

### Feature E: Sequitur training backbone

**What it adds**: replaces the beam+MDL training loop with Sequitur (Nevill-Manning &
Witten, 1997) for grammar induction. Sequitur builds a hierarchical CFG from a sequence
by replacing repeated digrams — O(n) time, deterministic, globally optimal under the
digram substitution criterion.

The SPMA beam search is kept at inference only. The pipeline becomes:

```
Train:  sequences → Sequitur → Grammar (CFG rules as Old patterns)
Infer:  New sequence + Grammar → SPMA beam search → Alignment + E cost
```

**What this resolves**:
- Cold-start n-gram miner: replaced by Sequitur's exact digram counting
- Greedy MDL: replaced by globally optimal digram substitution
- Training speed: O(n × cycles × K) → O(n) deterministic
- Determinism: Sequitur is fully deterministic (no HashMap iteration ordering issues)

**What this does not change**:
- Alignment table (beam search unchanged)
- E cost semantics (still bits of uncovered New symbols)
- Anomaly detection logic

**Implementation cost**: high. Sequitur is a separate algorithm that needs to be
implemented or integrated as a dependency. The multi-sequence case requires concatenating
sequences with separator symbols and handling cross-boundary rule suppression. The
resulting CFG rules use a different symbol ID space than the current interner.

The separator-boundary problem: Sequitur mines the concatenated sequence globally. A rule
`[A B]` learned at position 1000 might span a separator between sequence 47 and sequence
48 if A and B happen to be adjacent across that boundary. Mitigation: use a unique
separator symbol not in the alphabet, and post-filter any rule that contains a separator.
This is correct but requires careful implementation.

**Dependency**: grammar spec (Sequitur rules must serialize as `PatternRecord`s in the
same format). Does not depend on IC/OC or gaps.

**ROI**: high for production deployment (training speed, determinism). Lower priority than
features A-C for correctness/capability reasons. Recommended after benchmarking
establishes that training speed is a bottleneck.

---

## Recommended implementation order

```
Phase 1 — Foundation (no algorithmic change, high ROI)
  1a. Grammar spec v2 schema — bump serialization format, schema only
  1b. Structured Alignment struct — surface existing beam data, add Display
  1c. Calibrated E-score — store E distribution at training, emit percentile

Phase 2 — Core capability gap
  2a. Non-contiguous patterns (Gap) — requires Phase 1 schema
  2b. Internal benchmark on real EDF sequences — validate detection rate

Phase 3 — Full SP theory (after benchmark confirms competitive detection)
  3a. IC/OC symbol roles — requires Phase 1 + structured alignment
  3b. Sequitur backbone — optional, if training speed is a bottleneck

Phase 4 — Reserved
  4a. Pattern variables / unification — only if Phase 3 proves insufficient
```

Phase 1 has no risk: it is schema work and output enrichment with no behavior change.
Phase 2a (gaps) is the highest-ROI algorithmic change for the EDF use case.
Phase 2b (benchmark) is a decision gate — if detection rate is not competitive, stop and
diagnose before adding more features. If it is competitive, Phase 3 is the path to making
SPMA a general symbolic reasoning system, not just a specialized anomaly detector.

---

## What good looks like at each phase end

**After Phase 1**: a production-deployable library. Stable serialization format, rich
inference output (structured alignment table, anomaly percentile), no breaking changes.

**After Phase 2a**: detection of non-adjacent fault event co-occurrences. The most common
class of industrial fault pattern — "event A eventually leads to event B" — becomes
learnable.

**After Phase 2b**: empirical confirmation that SPMA's detection rate is competitive with
Drain + Isolation Forest on a real dataset. Without this, Phases 3 and 4 are building on
an unvalidated foundation.

**After Phase 3a**: full Wolff SP theory implementation. Patterns can bracket themselves
with identity codes, enabling structural analogy and true hierarchical composition.
SPMA is no longer a specialized log anomaly detector — it is a general symbolic sequence
understanding system with anomaly detection as one application.

**After Phase 3b**: training is O(n), deterministic, globally optimal. Suitable for online
retraining on streaming fault data without the current cycle-convergence overhead.
