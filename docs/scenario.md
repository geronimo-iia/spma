Beam correctness

Contiguous pattern exact match — all symbols covered, e_cost=0
Contiguous pattern partial match — tail unmatched contributes to E
Gap pattern match within window — flanking symbols covered, interior uncovered
Gap pattern rejected when skip > max — e_cost = full sequence cost
Gap pattern wrong order (B before A) — not matched
Two non-overlapping patterns cover disjoint spans — both rows in alignment
Single-symbol pattern matches twice in same sequence

Alignment construction
8. build_alignment row count matches distinct patterns in match_log
9. Gap cell inserted between non-adjacent match events
10. fully_matched true when all pattern symbols present, false when partial
11. unmatched_symbols returns only uncovered positions in order
12. Display output contains symbol names, E:, CD:, T:
13. Rows sorted by (level, first_new_pos)

Training
14. Train on repeated identical sequences → grammar non-empty, pattern covers sequence
15. Train on varied corpus → most frequent bigram appears as pattern
16. atom_costs: rare symbol costs more than frequent symbol
17. Gap pattern induced from corpus with varying middle element (TRIP/X/RESTORATION)
18. Multi-level: level-1 pattern induced when level-0 patterns co-occur consistently

Calibrated score
19. Train on 10× identical → all training seqs infer with e_norm=0.0
20. anomaly_percentile=0.0 for perfectly covered training sequence
21. anomaly_percentile>0.0 for novel sequence after training
22. is_anomaly=false for known sequence (default threshold=0.0, e_norm=0.0)
23. is_anomaly=true for fully unknown sequence
24. level_costs.len() == grammar.levels.len() after infer

Gap matching end-to-end
25. TRIP/X(varies)/RESTORATION corpus → learns gap pattern, infers TRIP/Y/RESTORATION with e_norm<1.0
26. TRIP/A/B/RESTORATION (gap=2, exceeds max=1) → not covered by gap pattern
27. RESTORATION/TRIP (wrong order) → is_anomaly=true

Edge cases
28. Empty sequence → no panic, e_cost=0
29. Single-symbol sequence → returns result, not panic
30. Infer with unknown symbol → e_cost>0, no panic
31. Train on single sequence (below min_freq threshold) → grammar may be empty, no panic
32. beam_k=1 → still returns a result

Regression
33. After adding gap patterns, contiguous-pattern inference still correct
34. e_norm consistent: infer same seq twice → same result

That's 34 scenarios. Some overlap with existing unit tests — those can be thin integration wrappers. The ones not covered by any current test: 6, 14 (as integration), 17, 18, 25-27, 29-32, 34.
