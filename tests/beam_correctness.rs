use spma::beam::{beam_search, RawAlignment};
use spma::model::{GapConstraint, Pattern, SymbolRef};

fn make_costs(n: usize, default: f64) -> Vec<f64> {
    vec![default; n]
}

fn atom_ids(symbols: &[u32]) -> Vec<SymbolRef> {
    symbols.iter().map(|&id| SymbolRef::Atom(id)).collect()
}

fn best(new: &[u32], patterns: &[&Pattern], k: usize, costs: &[f64]) -> RawAlignment {
    let mut results = beam_search(new, patterns, k, costs);
    results.sort_by(|a, b| {
        b.cd.partial_cmp(&a.cd)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let ac = a.covered.iter().filter(|&&c| c).count();
                let bc = b.covered.iter().filter(|&&c| c).count();
                bc.cmp(&ac)
            })
    });
    results
        .into_iter()
        .next()
        .expect("beam_search returned empty")
}

// Scenario 1: contiguous_exact_match
// Pattern [A,B,C], new=[A,B,C] → covered=[T,T,T], e_cost==0.0, match_log.len()==3
#[test]
fn contiguous_exact_match() {
    let new = vec![0u32, 1, 2];
    let p0 = Pattern::new_contiguous(0, atom_ids(&[0, 1, 2]), 0);
    let old = vec![&p0];
    let costs = make_costs(3, 1.0);
    let r = best(&new, &old, 10, &costs);
    assert!(r.covered[0], "A must be covered");
    assert!(r.covered[1], "B must be covered");
    assert!(r.covered[2], "C must be covered");
    assert_eq!(r.e_cost, 0.0, "e_cost must be 0");
    assert_eq!(r.match_log.len(), 3, "match_log must have 3 events");
}

// Scenario 2: contiguous_partial_match
// Pattern [A,B], new=[A,B,C], costs A=1.0 B=1.0 C=2.0
// → covered[0..1]=true, covered[2]=false, e_cost==2.0, match_log.len()==2
#[test]
fn contiguous_partial_match() {
    let new = vec![0u32, 1, 2];
    let p0 = Pattern::new_contiguous(0, atom_ids(&[0, 1]), 0);
    let old = vec![&p0];
    let costs = vec![1.0f64, 1.0, 2.0];
    let r = best(&new, &old, 10, &costs);
    assert!(r.covered[0], "A must be covered");
    assert!(r.covered[1], "B must be covered");
    assert!(!r.covered[2], "C must not be covered");
    assert!(
        (r.e_cost - 2.0).abs() < 1e-10,
        "e_cost must be 2.0 (cost of C)"
    );
    assert_eq!(r.match_log.len(), 2, "match_log must have 2 events");
}

// Scenario 3: gap_match_within_window
// Pattern{symbols:[A,B], gaps:[GapConstraint{0,2}]}, new=[A,X,B]
// → covered[0]=T (A), covered[1]=F (X, gap interior), covered[2]=T (B)
// → e_cost==1.0, match_log.len()==2
#[test]
fn gap_match_within_window() {
    let new = vec![0u32, 1, 2]; // A=0, X=1, B=2
    let p0 = Pattern::new_with_gaps(0, atom_ids(&[0, 2]), vec![GapConstraint::up_to(2)], 0);
    let old = vec![&p0];
    let costs = make_costs(3, 1.0);
    let r = best(&new, &old, 10, &costs);
    assert!(r.covered[0], "A at new[0] must be covered");
    assert!(
        !r.covered[1],
        "X at new[1] must not be covered (gap interior)"
    );
    assert!(r.covered[2], "B at new[2] must be covered");
    assert!(
        (r.e_cost - 1.0).abs() < 1e-10,
        "e_cost must be 1.0 (cost of X)"
    );
    assert_eq!(
        r.match_log.len(),
        2,
        "match_log must have 2 events (A and B only)"
    );
}

// Scenario 4: gap_rejected_when_skip_exceeds_max
// Pattern{symbols:[A,B], gaps:[GapConstraint{0,2}]}, new=[A,X,Y,Z,B]
// skip = 4-0-1 = 3 > max=2 → A and B not both covered by gap match
#[test]
fn gap_rejected_when_skip_exceeds_max() {
    let new = vec![0u32, 1, 2, 3, 4]; // A=0, X=1, Y=2, Z=3, B=4
    let p0 = Pattern::new_with_gaps(0, atom_ids(&[0, 4]), vec![GapConstraint::up_to(2)], 0);
    let old = vec![&p0];
    let costs = make_costs(5, 1.0);
    let r = best(&new, &old, 10, &costs);
    assert!(
        !(r.covered[0] && r.covered[4]),
        "A and B must not both be covered: gap skip=3 exceeds max=2"
    );
}

// Scenario 5: gap_wrong_order
// Pattern{symbols:[A,B], gaps:[GapConstraint{0,2}]}, new=[B,X,A]
// Pattern requires A before B; B is at new[0], A at new[2]
// → assert NOT (covered[0] && covered[2]) via gap pattern
#[test]
fn gap_wrong_order() {
    let new = vec![2u32, 1, 0]; // B=2 at new[0], X=1 at new[1], A=0 at new[2]
    let p0 = Pattern::new_with_gaps(0, atom_ids(&[0, 2]), vec![GapConstraint::up_to(2)], 0);
    let old = vec![&p0];
    let costs = make_costs(3, 1.0);
    let r = best(&new, &old, 10, &costs);
    assert!(
        !(r.covered[0] && r.covered[2]),
        "B(new[0]) and A(new[2]) must not both be covered: wrong order for gap pattern A→B"
    );
}

// Scenario 6: two_non_overlapping_patterns
// Pattern P0=[A,B], Pattern P1=[C,D], new=[A,B,C,D]
// → e_cost==0.0, all covered, match_log has events from both old_idx=0 and old_idx=1
#[test]
fn two_non_overlapping_patterns() {
    let new = vec![0u32, 1, 2, 3]; // A=0, B=1, C=2, D=3
    let p0 = Pattern::new_contiguous(0, atom_ids(&[0, 1]), 0);
    let p1 = Pattern::new_contiguous(1, atom_ids(&[2, 3]), 0);
    let old = vec![&p0, &p1];
    let costs = make_costs(4, 1.0);
    let r = best(&new, &old, 20, &costs);
    assert_eq!(r.e_cost, 0.0, "e_cost must be 0.0: all symbols covered");
    assert!(
        r.covered.iter().all(|&c| c),
        "all positions must be covered"
    );
    let has_p0 = r.match_log.iter().any(|e| e.old_idx == 0);
    let has_p1 = r.match_log.iter().any(|e| e.old_idx == 1);
    assert!(has_p0, "match_log must contain events from old_idx=0 (P0)");
    assert!(has_p1, "match_log must contain events from old_idx=1 (P1)");
}

// Scenario 7: single_symbol_pattern_matches_twice
// Pattern [A], new=[A,X,A], costs A=1.0 X=2.0
// → covered[0]=T, covered[1]=F, covered[2]=T, e_cost==2.0, match_log.len()==2
#[test]
fn single_symbol_pattern_matches_twice() {
    let new = vec![0u32, 1, 0]; // A=0, X=1, A=0
    let p0 = Pattern::new_contiguous(0, atom_ids(&[0]), 0);
    let old = vec![&p0];
    let costs = vec![1.0f64, 2.0];
    let r = best(&new, &old, 10, &costs);
    assert!(r.covered[0], "A at new[0] must be covered");
    assert!(!r.covered[1], "X at new[1] must not be covered");
    assert!(r.covered[2], "A at new[2] must be covered");
    assert!(
        (r.e_cost - 2.0).abs() < 1e-10,
        "e_cost must be 2.0 (cost of X)"
    );
    assert_eq!(r.match_log.len(), 2, "match_log must have 2 events");
}
