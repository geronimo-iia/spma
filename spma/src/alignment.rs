use std::collections::HashMap;
use std::fmt;

use crate::beam::RawAlignment;
use crate::model::{Grammar, Pattern, SymbolRef};

// ── Cell ──────────────────────────────────────────────────────────────────────

pub struct Cell {
    pub old_pos: usize,
    pub new_pos: usize,
    pub content: String,
    pub is_gap: bool,
    pub gap_span: usize,
    pub cost: f64,
}

// ── AlignmentRow ──────────────────────────────────────────────────────────────

pub struct AlignmentRow {
    pub pattern_id: u32,
    pub pattern_label: String,
    pub level: usize,
    pub cells: Vec<Cell>,
    pub fully_matched: bool,
}

impl AlignmentRow {
    fn first_new_pos(&self) -> Option<usize> {
        self.cells.iter().find(|c| !c.is_gap).map(|c| c.new_pos)
    }
}

// ── Alignment ─────────────────────────────────────────────────────────────────

pub struct Alignment {
    pub new_symbols: Vec<String>,
    pub rows: Vec<AlignmentRow>,
    pub covered: Vec<bool>,
    pub e_cost: f64,
    pub cd: f64,
    pub level_costs: Vec<f64>,
}

impl Alignment {
    pub fn unmatched_symbols(&self) -> Vec<&str> {
        self.new_symbols
            .iter()
            .enumerate()
            .filter(|&(i, _)| !self.covered[i])
            .map(|(_, s)| s.as_str())
            .collect()
    }
}

// ── build_alignment ───────────────────────────────────────────────────────────

pub fn build_alignment(
    raw: &RawAlignment,
    new_names: &[&str],
    old_patterns: &[&Pattern],
    _grammar: &Grammar,
) -> Alignment {
    // Step 1: sort by (old_idx, new_pos)
    let mut sorted_log = raw.match_log.clone();
    sorted_log.sort_by_key(|e| (e.old_idx, e.new_pos));

    // Step 2: group by old_idx
    let mut by_old_idx: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, event) in sorted_log.iter().enumerate() {
        by_old_idx.entry(event.old_idx).or_default().push(i);
    }

    let mut rows: Vec<AlignmentRow> = Vec::new();

    let mut old_idxs: Vec<usize> = by_old_idx.keys().copied().collect();
    old_idxs.sort();

    for old_idx in old_idxs {
        let event_indices = &by_old_idx[&old_idx];
        let pat = old_patterns[old_idx];

        // Build content cells
        let mut cells: Vec<Cell> = event_indices
            .iter()
            .map(|&i| {
                let e = &sorted_log[i];
                let content = match pat.symbols[e.old_pos] {
                    SymbolRef::Atom(_) => new_names[e.new_pos].to_string(),
                    SymbolRef::Pattern(id) => format!("P{id}"),
                };
                Cell {
                    old_pos: e.old_pos,
                    new_pos: e.new_pos,
                    content,
                    is_gap: false,
                    gap_span: 0,
                    cost: e.cost,
                }
            })
            .collect();

        cells.sort_by_key(|c| c.new_pos);

        // Insert gap cells after building all content cells
        if !pat.gaps.is_empty() {
            let mut with_gaps: Vec<Cell> = Vec::with_capacity(cells.len() * 2);
            for i in 0..cells.len() {
                let cell = &cells[i];
                with_gaps.push(Cell {
                    old_pos: cell.old_pos,
                    new_pos: cell.new_pos,
                    content: cell.content.clone(),
                    is_gap: false,
                    gap_span: 0,
                    cost: cell.cost,
                });
                if i + 1 < cells.len() {
                    let next = &cells[i + 1];
                    if next.new_pos > cell.new_pos + 1 {
                        let gap_span = next.new_pos - cell.new_pos - 1;
                        with_gaps.push(Cell {
                            old_pos: cell.old_pos,
                            new_pos: cell.new_pos + 1,
                            content: format!("<{gap_span}>"),
                            is_gap: true,
                            gap_span,
                            cost: 0.0,
                        });
                    }
                }
            }
            cells = with_gaps;
        }

        let non_gap_count = cells.iter().filter(|c| !c.is_gap).count();
        let fully_matched = non_gap_count == pat.symbols.len();

        let pattern_label = format!("P{}", pat.id);

        rows.push(AlignmentRow {
            pattern_id: pat.id,
            pattern_label,
            level: pat.level as usize,
            cells,
            fully_matched,
        });
    }

    // Step 4: sort by (level, first_new_pos)
    rows.sort_by_key(|r| (r.level, r.first_new_pos().unwrap_or(usize::MAX)));

    Alignment {
        new_symbols: new_names.iter().map(|s| s.to_string()).collect(),
        rows,
        covered: raw.covered.clone(),
        e_cost: raw.e_cost,
        cd: raw.cd,
        level_costs: Vec::new(),
    }
}

// ── Display ───────────────────────────────────────────────────────────────────

impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let col_width = self
            .new_symbols
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .saturating_add(2)
            .max(4);

        let label_width = self
            .rows
            .iter()
            .map(|r| format!("{}(L{})", r.pattern_label, r.level).len())
            .max()
            .unwrap_or(0);

        // Header
        write!(f, "{:width$}  ", "", width = label_width)?;
        for sym in &self.new_symbols {
            write!(f, "{:<width$}", sym, width = col_width)?;
        }
        writeln!(f)?;

        // Rows
        for row in &self.rows {
            let label = format!("{}(L{})", row.pattern_label, row.level);
            write!(f, "{:<width$}  ", label, width = label_width)?;

            let cell_map: HashMap<usize, &str> = row
                .cells
                .iter()
                .map(|c| (c.new_pos, c.content.as_str()))
                .collect();

            for i in 0..self.new_symbols.len() {
                let content = cell_map.get(&i).copied().unwrap_or(".");
                write!(f, "{:<width$}", content, width = col_width)?;
            }
            writeln!(f)?;
        }

        write!(
            f,
            "---\nE: {:.1} bits   CD: {:.1} bits   T: {:.1} bits",
            self.e_cost,
            self.cd,
            self.e_cost + self.cd
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beam::{beam_search, MatchEvent, RawAlignment};
    use crate::intern::Interner;
    use crate::model::{GapConstraint, Pattern, SymbolIndex, SymbolRef};

    fn make_grammar(symbols: &[&str]) -> (Grammar, Vec<u32>) {
        let mut interner = Interner::new();
        let ids: Vec<u32> = symbols.iter().map(|s| interner.intern(s)).collect();
        (Grammar::new(interner), ids)
    }

    #[test]
    fn single_pattern_two_symbols_three_new() {
        // Pattern [A,B], new=[A,B,C] → 1 row, 2 non-gap cells, covered=[T,T,F]
        let (grammar, ids) = make_grammar(&["A", "B", "C"]);
        let (a, b, c) = (ids[0], ids[1], ids[2]);

        let pat = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
        let costs = vec![1.0, 1.0, 2.0];
        let new = vec![a, b, c];
        let old_refs = vec![&pat];
        let idx = SymbolIndex::build(&old_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>());
        let results = beam_search(&new, &old_refs, &idx, 5, &costs);
        let raw = &results[0];
        let alignment = build_alignment(raw, &["A", "B", "C"], &old_refs, &grammar);

        assert_eq!(alignment.rows.len(), 1);
        let row = &alignment.rows[0];
        let non_gap = row.cells.iter().filter(|c| !c.is_gap).count();
        assert_eq!(non_gap, 2);
        assert_eq!(alignment.covered, vec![true, true, false]);
        assert!((alignment.e_cost - 2.0).abs() < 1e-10);
    }

    #[test]
    fn rows_sorted_by_level_then_first_pos() {
        // Level-1 pattern starting at new[2], level-0 starting at new[0] → level-0 row first
        let mut interner = Interner::new();
        let a = interner.intern("A");
        let b = interner.intern("B");
        let c = interner.intern("C");
        let d = interner.intern("D");
        let grammar = Grammar::new(interner);

        let pat0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(c), SymbolRef::Atom(d)], 1);
        let pat1 = Pattern::new_contiguous(1, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
        let costs = vec![1.0; 4];
        let new = vec![a, b, c, d];
        let old_refs = vec![&pat0, &pat1];
        let idx = SymbolIndex::build(&old_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>());
        let results = beam_search(&new, &old_refs, &idx, 20, &costs);
        let raw = &results[0];
        let alignment = build_alignment(raw, &["A", "B", "C", "D"], &old_refs, &grammar);

        assert_eq!(alignment.rows.len(), 2);
        // Level-0 row (pat1=[A,B]) must sort before level-1 row (pat0=[C,D])
        assert_eq!(alignment.rows[0].level, 0);
        assert_eq!(alignment.rows[1].level, 1);
    }

    #[test]
    fn unmatched_symbols_returns_uncovered_in_order() {
        let (grammar, ids) = make_grammar(&["A", "B", "C"]);
        let (a, _b, c) = (ids[0], ids[1], ids[2]);

        let pat = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(c)], 0);
        // Contiguous [A,C] won't match [A,B,C] since B is in between
        let costs = vec![1.0, 1.0, 1.0];
        let new = vec![a, ids[1], c];
        let old_refs = vec![&pat];
        let idx = SymbolIndex::build(&old_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>());
        let results = beam_search(&new, &old_refs, &idx, 5, &costs);
        let raw = &results[0];
        let alignment = build_alignment(raw, &["A", "B", "C"], &old_refs, &grammar);

        // A and C can't be matched together contiguously; at most A is matched alone
        // Check that unmatched_symbols only returns names where covered[i] == false
        let unmatched = alignment.unmatched_symbols();
        for name in &unmatched {
            let pos = ["A", "B", "C"].iter().position(|s| s == name).unwrap();
            assert!(!alignment.covered[pos]);
        }
        for (i, covered) in alignment.covered.iter().enumerate() {
            if *covered {
                assert!(!unmatched.contains(&["A", "B", "C"][i]));
            }
        }
    }

    #[test]
    fn display_contains_symbols_and_e_marker() {
        let (grammar, ids) = make_grammar(&["A", "B"]);
        let (a, b) = (ids[0], ids[1]);

        let pat = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
        let costs = vec![1.0, 1.0];
        let new = vec![a, b];
        let old_refs = vec![&pat];
        let idx = SymbolIndex::build(&old_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>());
        let results = beam_search(&new, &old_refs, &idx, 5, &costs);
        let raw = &results[0];
        let alignment = build_alignment(raw, &["A", "B"], &old_refs, &grammar);

        let s = alignment.to_string();
        assert!(s.contains("A"), "display must contain 'A'");
        assert!(s.contains("B"), "display must contain 'B'");
        assert!(s.contains("E:"), "display must contain 'E:'");
        assert!(s.contains("CD:"), "display must contain 'CD:'");
        assert!(s.contains("T:"), "display must contain 'T:'");
    }

    #[test]
    fn gap_cell_inserted_between_non_adjacent_matches() {
        let mut interner = Interner::new();
        let a_id = interner.intern("A");
        let b_id = interner.intern("B");
        let grammar = Grammar::new(interner);

        let pat = Pattern::new_with_gaps(
            0,
            vec![SymbolRef::Atom(a_id), SymbolRef::Atom(b_id)],
            vec![GapConstraint::up_to(2)],
            0,
        );
        let old_refs = vec![&pat];

        let raw = RawAlignment {
            match_log: vec![
                MatchEvent {
                    old_idx: 0,
                    old_pos: 0,
                    new_pos: 0,
                    cost: 1.0,
                },
                MatchEvent {
                    old_idx: 0,
                    old_pos: 1,
                    new_pos: 2,
                    cost: 1.0,
                },
            ],
            covered: vec![true, false, true],
            e_cost: 1.0,
            cd: 2.0,
        };

        let alignment = build_alignment(&raw, &["A", "X", "B"], &old_refs, &grammar);

        assert_eq!(alignment.rows.len(), 1);
        let row = &alignment.rows[0];
        assert_eq!(row.cells.len(), 3);
        assert_eq!(row.cells[0].new_pos, 0);
        assert_eq!(row.cells[0].content, "A");
        assert!(!row.cells[0].is_gap);
        assert_eq!(row.cells[1].new_pos, 1);
        assert_eq!(row.cells[1].content, "<1>");
        assert!(row.cells[1].is_gap);
        assert_eq!(row.cells[1].gap_span, 1);
        assert_eq!(row.cells[2].new_pos, 2);
        assert_eq!(row.cells[2].content, "B");
        assert!(!row.cells[2].is_gap);
    }

    // Scenarios 8-15: migrated from tests/alignment_construction.rs

    #[test]
    fn scenario8_row_count_matches_distinct_patterns() {
        let (grammar, ids) = make_grammar(&["A", "B", "C", "D"]);
        let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
        let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
        let p1 = Pattern::new_contiguous(1, vec![SymbolRef::Atom(c), SymbolRef::Atom(d)], 0);
        let costs = vec![1.0; 4];
        let new = vec![a, b, c, d];
        let old_refs = vec![&p0, &p1];
        let idx = SymbolIndex::build(&old_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>());
        let results = beam_search(&new, &old_refs, &idx, 20, &costs);
        let alignment = build_alignment(&results[0], &["A", "B", "C", "D"], &old_refs, &grammar);
        assert_eq!(alignment.rows.len(), 2);
        let ids_used: std::collections::HashSet<u32> =
            alignment.rows.iter().map(|r| r.pattern_id).collect();
        assert_eq!(ids_used.len(), 2);
    }

    #[test]
    fn scenario9_gap_cell_inserted_between_non_adjacent_events() {
        let mut interner = Interner::new();
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
                MatchEvent {
                    old_idx: 0,
                    old_pos: 0,
                    new_pos: 0,
                    cost: 1.0,
                },
                MatchEvent {
                    old_idx: 0,
                    old_pos: 1,
                    new_pos: 2,
                    cost: 1.0,
                },
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
        assert!(row.cells[1].is_gap);
        assert_eq!(row.cells[1].content, "<1>");
        assert_eq!(row.cells[1].gap_span, 1);
        assert_eq!(row.cells[2].content, "B");
        assert!(!row.cells[2].is_gap);
    }

    #[test]
    fn scenario10_fully_matched_true_and_false() {
        let (grammar, ids) = make_grammar(&["A", "B", "C"]);
        let (a, b, c) = (ids[0], ids[1], ids[2]);
        let p0 = Pattern::new_contiguous(
            0,
            vec![SymbolRef::Atom(a), SymbolRef::Atom(b), SymbolRef::Atom(c)],
            0,
        );
        let costs = vec![1.0; 3];
        let new = vec![a, b, c];
        let old_refs = vec![&p0];
        let idx = SymbolIndex::build(&old_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>());
        let results = beam_search(&new, &old_refs, &idx, 10, &costs);
        let alignment = build_alignment(&results[0], &["A", "B", "C"], &old_refs, &grammar);
        assert!(
            alignment.rows[0].fully_matched,
            "all 3 matched → fully_matched true"
        );

        let raw_partial = RawAlignment {
            match_log: vec![
                MatchEvent {
                    old_idx: 0,
                    old_pos: 0,
                    new_pos: 0,
                    cost: 1.0,
                },
                MatchEvent {
                    old_idx: 0,
                    old_pos: 1,
                    new_pos: 1,
                    cost: 1.0,
                },
            ],
            covered: vec![true, true, false],
            e_cost: 1.0,
            cd: 2.0,
        };
        let alignment2 = build_alignment(&raw_partial, &["A", "B", "C"], &old_refs, &grammar);
        assert!(
            !alignment2.rows[0].fully_matched,
            "only 2 of 3 matched → fully_matched false"
        );
    }

    #[test]
    fn scenario11_unmatched_symbols_in_order() {
        let mut interner = Interner::new();
        let a = interner.intern("A");
        let _ = interner.intern("B");
        let c = interner.intern("C");
        let _ = interner.intern("D");
        let grammar = Grammar::new(interner);
        let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(c)], 0);
        let old_refs = vec![&p0];
        let raw = RawAlignment {
            match_log: vec![MatchEvent {
                old_idx: 0,
                old_pos: 0,
                new_pos: 0,
                cost: 1.0,
            }],
            covered: vec![true, false, false, false],
            e_cost: 3.0,
            cd: 1.0,
        };
        let alignment = build_alignment(&raw, &["A", "B", "C", "D"], &old_refs, &grammar);
        let unmatched = alignment.unmatched_symbols();
        assert!(!unmatched.contains(&"A"));
        assert!(unmatched.contains(&"B"));
        assert!(unmatched.contains(&"C"));
        assert!(unmatched.contains(&"D"));
        let b_pos = unmatched.iter().position(|&s| s == "B").unwrap();
        let c_pos = unmatched.iter().position(|&s| s == "C").unwrap();
        let d_pos = unmatched.iter().position(|&s| s == "D").unwrap();
        assert!(b_pos < c_pos);
        assert!(c_pos < d_pos);
    }

    #[test]
    fn scenario12_display_contains_required_markers() {
        let (grammar, ids) = make_grammar(&["A", "B", "C", "D"]);
        let (a, b, c, d) = (ids[0], ids[1], ids[2], ids[3]);
        let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
        let p1 = Pattern::new_contiguous(1, vec![SymbolRef::Atom(c), SymbolRef::Atom(d)], 0);
        let costs = vec![1.0; 4];
        let new = vec![a, b, c, d];
        let old_refs = vec![&p0, &p1];
        let idx = SymbolIndex::build(&old_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>());
        let results = beam_search(&new, &old_refs, &idx, 20, &costs);
        let alignment = build_alignment(&results[0], &["A", "B", "C", "D"], &old_refs, &grammar);
        let s = alignment.to_string();
        assert!(s.contains('A'));
        assert!(s.contains("E:"));
        assert!(s.contains("CD:"));
        assert!(s.contains("T:"));
        let empty = Alignment {
            new_symbols: vec!["X".to_string()],
            rows: vec![],
            covered: vec![false],
            e_cost: 1.0,
            cd: 0.0,
            level_costs: vec![],
        };
        let _ = empty.to_string();
    }

    #[test]
    fn scenario13_rows_sorted_by_level_then_first_new_pos() {
        let mut interner = Interner::new();
        let a = interner.intern("A");
        let b = interner.intern("B");
        let c = interner.intern("C");
        let d = interner.intern("D");
        let grammar = Grammar::new(interner);
        let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(c), SymbolRef::Atom(d)], 1);
        let p1 = Pattern::new_contiguous(1, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
        let costs = vec![1.0; 4];
        let new = vec![a, b, c, d];
        let old_refs = vec![&p0, &p1];
        let idx = SymbolIndex::build(&old_refs.iter().map(|p| (*p).clone()).collect::<Vec<_>>());
        let results = beam_search(&new, &old_refs, &idx, 20, &costs);
        let alignment = build_alignment(&results[0], &["A", "B", "C", "D"], &old_refs, &grammar);
        assert_eq!(alignment.rows.len(), 2);
        assert_eq!(alignment.rows[0].level, 0);
        assert_eq!(alignment.rows[1].level, 1);
        assert_eq!(alignment.rows[0].pattern_id, 1);
        assert_eq!(alignment.rows[1].pattern_id, 0);
    }

    #[test]
    fn scenario14_display_footer_shows_zero_e_cost_for_fully_covered() {
        let mut spma = crate::engine::Spma::new(10);
        let c: Vec<Vec<&str>> = vec![vec!["TRIP", "OPEN", "RESTORE"]; 20];
        spma.train(&c);
        let result = spma.infer(&["TRIP", "OPEN", "RESTORE"]);
        assert!(result.e_cost < 1e-10);
        let s = format!("{}", result.alignment);
        assert!(s.contains("E: 0.0 bits") || s.contains("E: -0.0 bits"));
    }

    #[test]
    fn scenario15_unmatched_symbols_empty_when_fully_covered() {
        let mut spma = crate::engine::Spma::new(10);
        let c: Vec<Vec<&str>> = vec![vec!["TRIP", "OPEN", "RESTORE"]; 20];
        spma.train(&c);
        let result = spma.infer(&["TRIP", "OPEN", "RESTORE"]);
        assert!(result.alignment.unmatched_symbols().is_empty());
    }
}
