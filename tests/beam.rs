use spma::*;

#[test]
fn test_v3_beam_search_alignment() {
    let mut interner = spma::Interner::new();
    let the_id = interner.intern("the");
    let cat_id = interner.intern("cat");
    let sat_id = interner.intern("sat");
    let on_id = interner.intern("on");
    let mat_id = interner.intern("mat");
    let dog_id = interner.intern("dog");

    let n_syms = interner.len();
    let costs: Vec<f64> = vec![2.0; n_syms];

    let old: Vec<Vec<u32>> = vec![
        vec![the_id],
        vec![cat_id],
        vec![sat_id],
        vec![on_id],
        vec![mat_id],
    ];

    let new_full = vec![the_id, cat_id, sat_id, on_id, the_id, mat_id];
    let results = spma::beam_search(&new_full, &old, 10, &costs);
    assert!(!results.is_empty(), "beam search returned no alignments");

    let best = &results[0];
    assert!(
        best.covered_new.iter().all(|&c| c),
        "expected full coverage, got {:?}",
        best.covered_new
    );
    assert!(best.cd > 0.0, "expected CD > 0, got {}", best.cd);
    assert_eq!(best.alignment_type, spma::AlignmentType::FullA);

    let new_partial = vec![the_id, dog_id, sat_id, on_id, the_id, mat_id];
    let results2 = spma::beam_search(&new_partial, &old, 10, &costs);
    assert!(!results2.is_empty());
    let best2 = &results2[0];
    assert!(!best2.covered_new[1], "dog should be uncovered");
    let covered_count = best2.covered_new.iter().filter(|&&c| c).count();
    assert_eq!(covered_count, 5, "expected 5/6 covered");
    assert!(best2.cd >= 0.0, "expected CD >= 0, got {}", best2.cd);
    assert!(best2.t >= best.t, "partial T should be >= full T");
}

#[test]
fn beam_search_partial_coverage_correct_cd() {
    let mut i = Interner::new();
    let a = i.intern("A");
    let b = i.intern("B");
    let c = i.intern("C");
    let costs = vec![2.0, 3.0, 4.0];
    let old = vec![vec![a, b]];
    let results = spma::beam_search(&[a, b, c], &old, 10, &costs);
    let best = &results[0];
    assert!(best.covered_new[0], "A should be covered");
    assert!(best.covered_new[1], "B should be covered");
    assert!(!best.covered_new[2], "C should not be covered");
    assert!((best.cd - 5.0).abs() < 1e-9, "CD={}", best.cd);
    assert!((best.e - 4.0).abs() < 1e-9, "E={}", best.e);
}

#[test]
fn beam_search_multi_symbol_old_pattern() {
    let mut i = Interner::new();
    let a = i.intern("A");
    let b = i.intern("B");
    let c = i.intern("C");
    let costs = vec![1.0, 1.0, 1.0];
    let old = vec![vec![a, b, c]];
    let results = spma::beam_search(&[a, b, c], &old, 5, &costs);
    let best = &results[0];
    assert!(best.covered_new.iter().all(|&c| c), "all positions should be covered");
    assert_eq!(best.alignment_type, spma::AlignmentType::FullA);
}

#[test]
fn beam_search_alignment_type_full_b_partial_new() {
    let mut i = Interner::new();
    let a = i.intern("A");
    let b = i.intern("B");
    let costs = vec![1.0, 1.0];
    let old = vec![vec![a]];
    let results = spma::beam_search(&[a, b], &old, 5, &costs);
    let best = &results[0];
    assert!(best.covered_new[0]);
    assert!(!best.covered_new[1]);
    assert_eq!(best.alignment_type, spma::AlignmentType::FullB);
}

#[test]
fn beam_search_monotonic_order_enforced() {
    let mut i = Interner::new();
    let a = i.intern("A");
    let b = i.intern("B");
    let costs = vec![1.0, 1.0];
    let old = vec![vec![a, b]];
    let results = spma::beam_search(&[b, a], &old, 5, &costs);
    let best = &results[0];
    let covered_count = best.covered_new.iter().filter(|&&c| c).count();
    assert!(
        covered_count <= 1,
        "out-of-order sequence should not produce full coverage, got {covered_count}"
    );
}

// ── write_alignment_table ─────────────────────────────────────────────────

#[test]
fn write_alignment_table_empty_pattern_writes_nothing() {
    let interner = Interner::new();
    let empty_pat = Pattern::new(vec![], 0);
    let alignment = spma::beam_search(&[], &[], 5, &[]);
    let mut out = String::new();
    if let Some(best) = alignment.into_iter().next() {
        spma::write_alignment_table(&mut out, &empty_pat, &best, &[], &interner);
    }
    assert!(out.is_empty());
}

#[test]
fn write_alignment_table_no_old_patterns() {
    let mut interner = Interner::new();
    let a = interner.intern("A");
    let b = interner.intern("B");
    let new_syms = vec![Symbol::new(a), Symbol::new(b)];
    let new_pat = Pattern::new(new_syms, 1);
    let costs = vec![1.0, 1.0];
    let results = spma::beam_search(&[a, b], &[], 5, &costs);
    let best = results.into_iter().next().unwrap();
    let mut out = String::new();
    spma::write_alignment_table(&mut out, &new_pat, &best, &[], &interner);
    assert!(out.contains("New:"), "should contain New row");
    assert!(out.contains('A') || out.contains('B'), "should contain symbol names");
}

#[test]
fn write_alignment_table_multi_row_contains_old_labels() {
    let mut interner = Interner::new();
    let a = interner.intern("A");
    let b = interner.intern("B");
    let costs = vec![2.0, 2.0];
    let old = vec![vec![a], vec![b]];
    let new_ids = vec![a, b];
    let results = spma::beam_search(&new_ids, &old, 10, &costs);
    let best = results.into_iter().next().unwrap();

    let new_syms: Vec<Symbol> = new_ids.iter().map(|&id| Symbol::new(id)).collect();
    let new_pat = Pattern::new(new_syms, 0);
    let old_pats: Vec<Pattern> = old
        .iter()
        .map(|ids| Pattern::new(ids.iter().map(|&id| Symbol::new(id)).collect(), 0))
        .collect();

    let mut out = String::new();
    spma::write_alignment_table(&mut out, &new_pat, &best, &old_pats, &interner);
    assert!(out.contains("New:"));
    assert!(
        out.contains("Old1:") || out.contains("Old2:"),
        "multi-row should have Old labels, got:\n{out}"
    );
    assert!(out.contains("Matched:"), "should contain stats line");
}

// ── inter-pattern ordering (Issue #5 partial fix) ─────────────────────────

#[test]
fn inter_pattern_order_correct_order_fully_covered() {
    // old = [[A, B], [C, D]], new = [A, B, C, D] → E = 0
    let new = vec![0u32, 1, 2, 3];
    let old = vec![vec![0u32, 1u32], vec![2u32, 3u32]];
    let costs = vec![1.0, 1.0, 1.0, 1.0];
    let results = spma::beam_search(&new, &old, 20, &costs);
    let best = &results[0];
    assert_eq!(best.e, 0.0, "correct order: all symbols must be covered");
    assert!(best.covered_new.iter().all(|&c| c));
}

#[test]
fn inter_pattern_order_interleaved_rejected() {
    // old = [[A, B], [C, D]], new = [A, C, B, D]
    // Span contiguity (Issue #3) blocks both patterns from fully matching → E > 0.
    let new = vec![0u32, 2, 1, 3];
    let old = vec![vec![0u32, 1u32], vec![2u32, 3u32]];
    let costs = vec![1.0, 1.0, 1.0, 1.0];
    let results = spma::beam_search(&new, &old, 20, &costs);
    let best = &results[0];
    assert!(
        best.e > 0.0,
        "interleaved pattern: must not fully cover [A,C,B,D] with [A,B] and [C,D]"
    );
}
