use spma::alignment::{build_alignment, Alignment};
use spma::beam::{beam_search, MatchEvent, RawAlignment};
use spma::model::{GapConstraint, Grammar, Pattern, SymbolRef};

fn make_grammar(symbols: &[&str]) -> (spma::model::Grammar, Vec<u32>) {
    let mut interner = spma::Interner::new();
    let ids: Vec<u32> = symbols.iter().map(|s| interner.intern(s)).collect();
    (spma::model::Grammar::new(interner), ids)
}

// Scenario 8
#[test]
fn row_count_matches_distinct_patterns() {
    let (grammar, ids) = make_grammar(&["A", "B", "C", "D"]);
    let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);

    let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
    let p1 = Pattern::new_contiguous(1, vec![SymbolRef::Atom(c), SymbolRef::Atom(d)], 0);
    let costs = vec![1.0; 4];
    let new = vec![a, b, c, d];
    let old_refs = vec![&p0, &p1];

    let results = beam_search(&new, &old_refs, 20, &costs);
    let raw = &results[0];
    let alignment = build_alignment(raw, &["A", "B", "C", "D"], &old_refs, &grammar);

    assert_eq!(alignment.rows.len(), 2);
    let ids_used: std::collections::HashSet<u32> =
        alignment.rows.iter().map(|r| r.pattern_id).collect();
    assert_eq!(ids_used.len(), 2, "each row must have a distinct pattern_id");
}

// Scenario 9
#[test]
fn gap_cell_inserted_between_non_adjacent_events() {
    let mut interner = spma::Interner::new();
    let a_id = interner.intern("A");
    let b_id = interner.intern("B");
    let _ = interner.intern("X");
    let grammar = Grammar::new(interner);

    let p0 = Pattern::new_with_gaps(
        0,
        vec![SymbolRef::Atom(a_id), SymbolRef::Atom(b_id)],
        vec![GapConstraint::up_to(2)],
        0,
    );
    let old_refs = vec![&p0];

    let raw = RawAlignment {
        match_log: vec![
            MatchEvent { old_idx: 0, old_pos: 0, new_pos: 0, cost: 1.0 },
            MatchEvent { old_idx: 0, old_pos: 1, new_pos: 2, cost: 1.0 },
        ],
        covered: vec![true, false, true],
        e_cost: 1.0,
        cd: 2.0,
    };

    let alignment = build_alignment(&raw, &["A", "X", "B"], &old_refs, &grammar);

    assert_eq!(alignment.rows.len(), 1);
    let row = &alignment.rows[0];
    assert_eq!(row.cells.len(), 3);

    assert_eq!(row.cells[0].content, "A");
    assert!(!row.cells[0].is_gap);
    assert_eq!(row.cells[0].new_pos, 0);

    assert!(row.cells[1].is_gap);
    assert_eq!(row.cells[1].content, "<1>");
    assert_eq!(row.cells[1].gap_span, 1);
    assert_eq!(row.cells[1].new_pos, 1);

    assert_eq!(row.cells[2].content, "B");
    assert!(!row.cells[2].is_gap);
    assert_eq!(row.cells[2].new_pos, 2);
}

// Scenario 10
#[test]
fn fully_matched_true_and_false() {
    let (grammar, ids) = make_grammar(&["A", "B", "C"]);
    let (a, b, c) = (ids[0], ids[1], ids[2]);

    let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(b), SymbolRef::Atom(c)], 0);
    let costs = vec![1.0; 3];
    let new = vec![a, b, c];
    let old_refs = vec![&p0];

    // Full match via beam_search
    let results = beam_search(&new, &old_refs, 10, &costs);
    let raw = &results[0];
    let alignment = build_alignment(raw, &["A", "B", "C"], &old_refs, &grammar);
    assert!(alignment.rows[0].fully_matched, "all 3 symbols matched → fully_matched must be true");

    // Partial match: manual RawAlignment with only 2 of 3 events
    let mut interner2 = spma::Interner::new();
    let a2 = interner2.intern("A");
    let b2 = interner2.intern("B");
    let _c2 = interner2.intern("C");
    let grammar2 = Grammar::new(interner2);

    let p0b = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a2), SymbolRef::Atom(b2), SymbolRef::Atom(_c2)], 0);
    let old_refs2 = vec![&p0b];

    let raw_partial = RawAlignment {
        match_log: vec![
            MatchEvent { old_idx: 0, old_pos: 0, new_pos: 0, cost: 1.0 },
            MatchEvent { old_idx: 0, old_pos: 1, new_pos: 1, cost: 1.0 },
        ],
        covered: vec![true, true, false],
        e_cost: 1.0,
        cd: 2.0,
    };

    let alignment2 = build_alignment(&raw_partial, &["A", "B", "C"], &old_refs2, &grammar2);
    assert!(!alignment2.rows[0].fully_matched, "only 2 of 3 symbols matched → fully_matched must be false");
}

// Scenario 11
#[test]
fn unmatched_symbols_in_order() {
    let mut interner = spma::Interner::new();
    let a = interner.intern("A");
    let _b = interner.intern("B");
    let c = interner.intern("C");
    let _d = interner.intern("D");
    let grammar = Grammar::new(interner);

    let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(c)], 0);
    let old_refs = vec![&p0];

    // Manual RawAlignment: only A covered at new[0]
    let raw = RawAlignment {
        match_log: vec![MatchEvent { old_idx: 0, old_pos: 0, new_pos: 0, cost: 1.0 }],
        covered: vec![true, false, false, false],
        e_cost: 3.0,
        cd: 1.0,
    };

    let alignment = build_alignment(&raw, &["A", "B", "C", "D"], &old_refs, &grammar);
    let unmatched = alignment.unmatched_symbols();

    // Covered symbols not in unmatched
    assert!(!unmatched.contains(&"A"), "A is covered, must not be in unmatched");

    // Uncovered symbols in unmatched
    assert!(unmatched.contains(&"B"), "B is uncovered, must be in unmatched");
    assert!(unmatched.contains(&"C"), "C is uncovered, must be in unmatched");
    assert!(unmatched.contains(&"D"), "D is uncovered, must be in unmatched");

    // Order: left-to-right in new_symbols
    let b_pos = unmatched.iter().position(|&s| s == "B").unwrap();
    let c_pos = unmatched.iter().position(|&s| s == "C").unwrap();
    let d_pos = unmatched.iter().position(|&s| s == "D").unwrap();
    assert!(b_pos < c_pos, "B must appear before C");
    assert!(c_pos < d_pos, "C must appear before D");
}

// Scenario 12
#[test]
fn display_contains_required_markers() {
    // Reuse scenario 8 setup
    let (grammar, ids) = make_grammar(&["A", "B", "C", "D"]);
    let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);

    let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
    let p1 = Pattern::new_contiguous(1, vec![SymbolRef::Atom(c), SymbolRef::Atom(d)], 0);
    let costs = vec![1.0; 4];
    let new = vec![a, b, c, d];
    let old_refs = vec![&p0, &p1];

    let results = beam_search(&new, &old_refs, 20, &costs);
    let raw = &results[0];
    let alignment = build_alignment(raw, &["A", "B", "C", "D"], &old_refs, &grammar);

    let s = alignment.to_string();
    assert!(s.contains('A'), "display must contain 'A'");
    assert!(s.contains("E:"), "display must contain 'E:'");
    assert!(s.contains("CD:"), "display must contain 'CD:'");
    assert!(s.contains("T:"), "display must contain 'T:'");

    // Empty rows alignment must not panic
    let empty_alignment = Alignment {
        new_symbols: vec!["X".to_string()],
        rows: vec![],
        covered: vec![false],
        e_cost: 1.0,
        cd: 0.0,
        level_costs: vec![],
    };
    let _ = empty_alignment.to_string();
}

// Scenario 13
#[test]
fn rows_sorted_by_level_then_first_new_pos() {
    let mut interner = spma::Interner::new();
    let a = interner.intern("A");
    let b = interner.intern("B");
    let c = interner.intern("C");
    let d = interner.intern("D");
    let grammar = Grammar::new(interner);

    // P0 = level=1, starts at new[2]; P1 = level=0, starts at new[0]
    let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(c), SymbolRef::Atom(d)], 1);
    let p1 = Pattern::new_contiguous(1, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
    let costs = vec![1.0; 4];
    let new = vec![a, b, c, d];
    let old_refs = vec![&p0, &p1];

    let results = beam_search(&new, &old_refs, 20, &costs);
    let raw = &results[0];
    let alignment = build_alignment(raw, &["A", "B", "C", "D"], &old_refs, &grammar);

    assert_eq!(alignment.rows.len(), 2);
    assert_eq!(alignment.rows[0].level, 0, "level-0 row must sort first");
    assert_eq!(alignment.rows[1].level, 1, "level-1 row must sort second");
    assert_eq!(alignment.rows[0].pattern_id, 1, "rows[0] must be P1 (level=0)");
    assert_eq!(alignment.rows[1].pattern_id, 0, "rows[1] must be P0 (level=1)");
}

// Scenario 14 — display_footer_shows_zero_e_cost_for_fully_covered
#[test]
fn display_footer_shows_zero_e_cost_for_fully_covered() {
    let mut spma = spma::Spma::new(10);
    let c: Vec<Vec<&str>> = vec![vec!["TRIP", "OPEN", "RESTORE"]; 20];
    spma.train(&c);

    let result = spma.infer(&["TRIP", "OPEN", "RESTORE"]);
    assert!(
        result.e_cost < 1e-10,
        "fully covered sequence: e_cost must be ~0, got {}",
        result.e_cost
    );

    let s = format!("{}", result.alignment);
    assert!(
        s.contains("E: 0.0 bits") || s.contains("E: -0.0 bits"),
        "display footer must show zero e_cost, got:\n{}",
        s
    );
}

// Scenario 15 — unmatched_symbols_empty_when_fully_covered
#[test]
fn unmatched_symbols_empty_when_fully_covered() {
    let mut spma = spma::Spma::new(10);
    let c: Vec<Vec<&str>> = vec![vec!["TRIP", "OPEN", "RESTORE"]; 20];
    spma.train(&c);

    let result = spma.infer(&["TRIP", "OPEN", "RESTORE"]);
    assert!(
        result.alignment.unmatched_symbols().is_empty(),
        "fully covered sequence must have no unmatched symbols"
    );
}
