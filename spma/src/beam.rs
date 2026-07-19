use crate::model::{Pattern, SymbolIndex};

// ── MatchEvent / MatchArena ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MatchEvent {
    pub old_idx: usize,
    pub old_pos: usize,
    pub new_pos: usize,
    pub cost: f64,
}

struct MatchNode {
    event: MatchEvent,
    parent: Option<u32>,
}

struct MatchArena {
    nodes: Vec<MatchNode>,
}

impl MatchArena {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn push(&mut self, event: MatchEvent, parent: Option<u32>) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(MatchNode { event, parent });
        idx
    }

    fn collect(&self, tail: Option<u32>) -> Vec<MatchEvent> {
        let mut out = Vec::new();
        let mut cur = tail;
        while let Some(i) = cur {
            let node = &self.nodes[i as usize];
            out.push(node.event.clone());
            cur = node.parent;
        }
        out.reverse();
        out
    }
}

// ── PartialAlignment ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct PartialAlignment {
    new_cursors: Vec<u16>, // pattern_idx → new_pos; u16::MAX = absent
    covered_new: [u64; 8], // bitmask; bit i = new[i] covered; covers up to 512 symbols
    new_len: usize,        // stored for finalize
    max_covered_new: usize,
    cd: f64,
    log_tail: Option<u32>,
}

impl PartialAlignment {
    fn new(new_len: usize, n_pats: usize) -> Self {
        // assert!, not debug_assert: sequence > 512 would silently set wrong bits
        assert!(
            new_len <= 512,
            "beam_search: sequence length {new_len} exceeds 512-symbol bitmask limit"
        );
        // Safety: beam_search returns early on empty input before calling new(),
        // so new_len == 0 never reaches this assert.
        Self {
            new_cursors: vec![u16::MAX; n_pats],
            covered_new: [0u64; 8],
            new_len,
            max_covered_new: 0,
            cd: 0.0,
            log_tail: None,
        }
    }

    fn extend_skip(&self) -> Self {
        self.clone()
    }

    fn extend_match(
        &self,
        old_idx: usize,
        old_pos: usize,
        new_pos: usize,
        symbol_cost: f64,
        arena: &mut MatchArena,
    ) -> Self {
        debug_assert!(
            new_pos < u16::MAX as usize,
            "new_pos {new_pos} overflows u16"
        );
        let mut next = self.clone();
        next.covered_new[new_pos / 64] |= 1u64 << (new_pos % 64);
        next.cd += symbol_cost;
        next.new_cursors[old_idx] = new_pos as u16;
        if new_pos > next.max_covered_new {
            next.max_covered_new = new_pos;
        }
        let event = MatchEvent {
            old_idx,
            old_pos,
            new_pos,
            cost: symbol_cost,
        };
        next.log_tail = Some(arena.push(event, self.log_tail));
        next
    }

    fn can_extend(
        &self,
        old_idx: usize,
        old_pos: usize,
        new_pos: usize,
        patterns: &[&Pattern],
    ) -> bool {
        let pat = patterns[old_idx];
        let prev_new_raw = self.new_cursors[old_idx];
        if prev_new_raw == u16::MAX {
            old_pos == 0 && new_pos >= self.max_covered_new
        } else {
            let prev_new = prev_new_raw as usize;
            if old_pos == 0 {
                // Fresh restart of same pattern — must not overlap prior matches.
                new_pos >= self.max_covered_new
            } else if pat.gaps.is_empty() {
                // Advancing within a contiguous pattern.
                new_pos == prev_new + 1
            } else {
                // Advancing within a gap pattern — check constraint.
                let gap = &pat.gaps[old_pos - 1];
                let skip = new_pos.saturating_sub(prev_new + 1);
                skip >= gap.min && skip <= gap.max
            }
        }
    }

    fn finalize(self, new: &[u32], costs: &[f64], arena: &MatchArena) -> RawAlignment {
        let e_cost: f64 = new
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.covered_new[i / 64] & (1u64 << (i % 64)) == 0)
            .map(|(_, &id)| costs[id as usize])
            .sum();
        let match_log = arena.collect(self.log_tail);
        // Reconstruct Vec<bool> for RawAlignment (consumed by alignment.rs)
        let covered: Vec<bool> = (0..self.new_len)
            .map(|i| self.covered_new[i / 64] & (1u64 << (i % 64)) != 0)
            .collect();
        RawAlignment {
            match_log,
            covered,
            e_cost,
            cd: self.cd,
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RawAlignment {
    pub match_log: Vec<MatchEvent>,
    pub covered: Vec<bool>,
    pub e_cost: f64,
    pub cd: f64,
}

// ── beam_search ───────────────────────────────────────────────────────────────

pub fn beam_search(
    new: &[u32],
    old: &[&Pattern],
    index: &SymbolIndex,
    beam_k: usize,
    costs: &[f64],
) -> Vec<RawAlignment> {
    if new.is_empty() {
        return vec![];
    }

    let mut arena = MatchArena::new();
    let mut candidates = vec![PartialAlignment::new(new.len(), old.len())];

    for (p, &sym) in new.iter().enumerate() {
        let sym_cost = costs[sym as usize];
        let mut next_candidates = Vec::with_capacity(candidates.len() * 2);

        for candidate in &candidates {
            next_candidates.push(candidate.extend_skip());

            let matches = index.get(sym);
            if !matches.is_empty() {
                for &(oi, q) in matches {
                    if candidate.can_extend(oi as usize, q as usize, p, old) {
                        next_candidates.push(candidate.extend_match(
                            oi as usize,
                            q as usize,
                            p,
                            sym_cost,
                            &mut arena,
                        ));
                    }
                }
            }
        }

        next_candidates.sort_by(|a, b| {
            b.cd.partial_cmp(&a.cd)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let b_ones: u32 = b.covered_new.iter().map(|w| w.count_ones()).sum();
                    let a_ones: u32 = a.covered_new.iter().map(|w| w.count_ones()).sum();
                    b_ones.cmp(&a_ones)
                })
                .then_with(|| a.covered_new.cmp(&b.covered_new))
        });
        next_candidates.truncate(beam_k);
        candidates = next_candidates;
    }

    let mut results: Vec<RawAlignment> = candidates
        .into_iter()
        .map(|c| c.finalize(new, costs, &arena))
        .collect();
    results.sort_by(|a, b| b.cd.partial_cmp(&a.cd).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Pattern, SymbolIndex, SymbolRef};

    #[test]
    fn bitmask_coverage_round_trip() {
        let mut mask = [0u64; 8];
        // set bits 0, 2, 5
        mask[0] |= 1u64 << 0;
        mask[0] |= 1u64 << 2;
        mask[0] |= 1u64 << 5;
        assert!(mask[0] & (1u64 << 0) != 0);
        assert!(mask[0] & (1u64 << 1) == 0);
        assert!(mask[0] & (1u64 << 2) != 0);
        assert!(mask[0] & (1u64 << 5) != 0);
        assert!(mask[0] & (1u64 << 6) == 0);
        let total: u32 = mask.iter().map(|w| w.count_ones()).sum();
        assert_eq!(total, 3);
        // test cross-word bit (bit 64 = word 1, bit 0)
        mask[1] |= 1u64 << 0;
        let total2: u32 = mask.iter().map(|w| w.count_ones()).sum();
        assert_eq!(total2, 4);
    }

    fn contiguous_pattern(id: u32, atoms: &[u32]) -> Pattern {
        Pattern::new_contiguous(id, atoms.iter().map(|&a| SymbolRef::Atom(a)).collect(), 0)
    }

    fn index_for(patterns: &[&Pattern]) -> SymbolIndex {
        let owned: Vec<Pattern> = patterns.iter().map(|p| (*p).clone()).collect();
        SymbolIndex::build(&owned)
    }

    #[test]
    fn test_arena_fork() {
        let mut arena = MatchArena::new();
        let root = arena.push(
            MatchEvent {
                old_idx: 0,
                old_pos: 0,
                new_pos: 0,
                cost: 1.0,
            },
            None,
        );
        let branch_a = arena.push(
            MatchEvent {
                old_idx: 0,
                old_pos: 1,
                new_pos: 1,
                cost: 2.0,
            },
            Some(root),
        );
        let branch_b = arena.push(
            MatchEvent {
                old_idx: 1,
                old_pos: 0,
                new_pos: 2,
                cost: 3.0,
            },
            Some(root),
        );

        let log_a = arena.collect(Some(branch_a));
        assert_eq!(log_a.len(), 2);
        assert_eq!(log_a[0].new_pos, 0);
        assert_eq!(log_a[1].new_pos, 1);
        assert_eq!(log_a[1].old_idx, 0);

        let log_b = arena.collect(Some(branch_b));
        assert_eq!(log_b.len(), 2);
        assert_eq!(log_b[0].new_pos, 0);
        assert_eq!(log_b[1].new_pos, 2);
        assert_eq!(log_b[1].old_idx, 1);
    }

    #[test]
    fn test_beam_empty_old() {
        let new = vec![0u32, 1, 2];
        let old: Vec<Pattern> = vec![];
        let old_refs: Vec<&Pattern> = old.iter().collect();
        let costs = vec![1.0, 2.0, 3.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 5, &costs);
        assert_eq!(results.len(), 1);
        assert!(results[0].covered.iter().all(|&c| !c));
        assert_eq!(results[0].cd, 0.0);
    }

    #[test]
    fn test_beam_single_match() {
        let new = vec![0u32, 1];
        let p0 = contiguous_pattern(0, &[0u32]);
        let old_refs = vec![&p0];
        let costs = vec![2.0, 3.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 5, &costs);
        assert!(!results.is_empty());
        let best = &results[0];
        assert!(best.covered[0]);
    }

    #[test]
    fn span_contiguity_non_contiguous_gap_not_covered() {
        // old = [[A, B]], new = [A, X, B]
        // B at new[2] must NOT be covered — gap between A(0) and B(2)
        let new = vec![0u32, 1, 2];
        let p0 = contiguous_pattern(0, &[0u32, 2u32]);
        let old_refs = vec![&p0];
        let costs = vec![1.0, 1.0, 1.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 10, &costs);
        let best = &results[0];
        assert!(
            !best.covered[2],
            "B at new[2] must not be covered: gap between A(new[0]) and B(new[2])"
        );
    }

    #[test]
    fn span_contiguity_contiguous_pair_both_covered() {
        // old = [[A, B]], new = [A, B, C] → A and B covered, C not
        let new = vec![0u32, 1, 2];
        let p0 = contiguous_pattern(0, &[0u32, 1u32]);
        let old_refs = vec![&p0];
        let costs = vec![1.0, 1.0, 1.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 10, &costs);
        let best = &results[0];
        assert!(best.covered[0], "A at new[0] should be covered");
        assert!(best.covered[1], "B at new[1] should be covered");
        assert!(!best.covered[2], "C at new[2] should not be covered");
    }

    #[test]
    fn inter_pattern_order_correct_order_fully_covered() {
        // old = [[A,B],[C,D]], new = [A,B,C,D] → e_cost == 0
        let new = vec![0u32, 1, 2, 3];
        let p0 = contiguous_pattern(0, &[0u32, 1u32]);
        let p1 = contiguous_pattern(1, &[2u32, 3u32]);
        let old_refs = vec![&p0, &p1];
        let costs = vec![1.0, 1.0, 1.0, 1.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 20, &costs);
        let best = &results[0];
        assert_eq!(
            best.e_cost, 0.0,
            "correct order: all symbols must be covered"
        );
        assert!(best.covered.iter().all(|&c| c));
    }

    #[test]
    fn inter_pattern_order_interleaved_rejected() {
        // old = [[A,B],[C,D]], new = [A,C,B,D] → interleaved, e_cost > 0
        let new = vec![0u32, 2, 1, 3];
        let p0 = contiguous_pattern(0, &[0u32, 1u32]);
        let p1 = contiguous_pattern(1, &[2u32, 3u32]);
        let old_refs = vec![&p0, &p1];
        let costs = vec![1.0, 1.0, 1.0, 1.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 20, &costs);
        let best = &results[0];
        assert!(
            best.e_cost > 0.0,
            "interleaved pattern: must not fully cover [A,C,B,D] with [A,B] and [C,D]"
        );
    }

    #[test]
    fn single_symbol_pattern_matches_twice() {
        // old = [[A]], new = [A, B, A]
        // Pattern [A] should cover new[0] and new[2]. E = cost(B).
        let new = vec![0u32, 1u32, 0u32];
        let p0 = contiguous_pattern(0, &[0u32]);
        let old_refs = vec![&p0];
        let costs = vec![1.0, 1.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 10, &costs);
        let best = &results[0];
        assert!(best.covered[0], "A at new[0] should be covered");
        assert!(!best.covered[1], "B at new[1] should not be covered");
        assert!(best.covered[2], "A at new[2] should be covered");
        assert_eq!(best.e_cost, 1.0);
    }

    fn gap_pattern(id: u32, atoms: &[u32], max_gap: usize) -> Pattern {
        assert_eq!(
            atoms.len(),
            2,
            "gap_pattern helper only supports 2-symbol patterns"
        );
        Pattern::new_with_gaps(
            id,
            atoms.iter().map(|&a| SymbolRef::Atom(a)).collect(),
            vec![crate::model::GapConstraint::up_to(max_gap)],
            0,
        )
    }

    #[test]
    fn gap_match_within_window_both_covered() {
        // Pattern [A,B] gap(0,2), new=[A,X,B] → skip=1 in [0,2] → both covered
        let new = vec![0u32, 1u32, 2u32]; // A=0, X=1, B=2
        let p0 = gap_pattern(0, &[0u32, 2u32], 2);
        let old_refs = vec![&p0];
        let costs = vec![1.0, 1.0, 1.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 10, &costs);
        let best = &results[0];
        assert!(best.covered[0], "A at new[0] must be covered");
        assert!(
            !best.covered[1],
            "X at new[1] must NOT be covered (gap interior)"
        );
        assert!(best.covered[2], "B at new[2] must be covered");
        assert!(
            (best.e_cost - 1.0).abs() < 1e-10,
            "e_cost must be cost(X)=1.0"
        );
    }

    #[test]
    fn gap_too_wide_not_covered() {
        // Pattern [A,B] gap(0,2), new=[A,X,Y,Z,B] → skip=3 > max=2 → B NOT covered
        let new = vec![0u32, 1u32, 2u32, 3u32, 4u32]; // A=0, X=1, Y=2, Z=3, B=4
        let p0 = gap_pattern(0, &[0u32, 4u32], 2);
        let old_refs = vec![&p0];
        let costs = vec![1.0; 5];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 10, &costs);
        let best = &results[0];
        assert!(
            !best.covered[4] || !best.covered[0],
            "gap too wide: A+B must not both be covered together"
        );
        // At minimum B at new[4] must not be covered as part of the gap pattern
        // since skip=3 exceeds max=2
        assert!(best.e_cost > 0.0, "e_cost must be > 0 when gap is too wide");
    }

    #[test]
    fn gap_wrong_order_not_covered() {
        // Pattern [A,B] gap(0,2), new=[B,A] → wrong order, neither covered together
        let new = vec![2u32, 0u32]; // B=2, A=0
        let p0 = gap_pattern(0, &[0u32, 2u32], 2);
        let old_refs = vec![&p0];
        let costs = vec![1.0; 3];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 10, &costs);
        let best = &results[0];
        // A at new[1] starts a fresh match (old_pos=0), B at new[0] can't follow
        // So at most one of A/B is covered (A alone from a single-symbol perspective)
        // But since pattern [A,B] has 2 symbols, both can't be matched out of order
        assert!(best.e_cost > 0.0, "wrong order: e_cost must be > 0");
    }

    #[test]
    fn match_log_records_events() {
        // old = [[A, B]], new = [A, B] → match_log has 2 events
        let new = vec![0u32, 1];
        let p0 = contiguous_pattern(0, &[0u32, 1u32]);
        let old_refs = vec![&p0];
        let costs = vec![1.0, 2.0];
        let idx = index_for(&old_refs);
        let results = beam_search(&new, &old_refs, &idx, 5, &costs);
        let best = &results[0];
        assert_eq!(best.match_log.len(), 2);
        assert_eq!(best.match_log[0].new_pos, 0);
        assert_eq!(best.match_log[1].new_pos, 1);
        assert_eq!(best.match_log[0].old_pos, 0);
        assert_eq!(best.match_log[1].old_pos, 1);
    }

    // Scenarios 1-7: integration-level beam correctness (migrated from tests/beam_correctness.rs)

    fn best_result(new: &[u32], patterns: &[&Pattern], k: usize, costs: &[f64]) -> RawAlignment {
        let idx = index_for(patterns);
        let mut results = beam_search(new, patterns, &idx, k, costs);
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

    #[test]
    fn scenario1_contiguous_exact_match() {
        let new = vec![0u32, 1, 2];
        let p0 = Pattern::new_contiguous(
            0,
            vec![SymbolRef::Atom(0), SymbolRef::Atom(1), SymbolRef::Atom(2)],
            0,
        );
        let old = vec![&p0];
        let costs = vec![1.0; 3];
        let r = best_result(&new, &old, 10, &costs);
        assert!(r.covered[0]);
        assert!(r.covered[1]);
        assert!(r.covered[2]);
        assert_eq!(r.e_cost, 0.0);
        assert_eq!(r.match_log.len(), 3);
    }

    #[test]
    fn scenario2_contiguous_partial_match() {
        let new = vec![0u32, 1, 2];
        let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(0), SymbolRef::Atom(1)], 0);
        let old = vec![&p0];
        let costs = vec![1.0f64, 1.0, 2.0];
        let r = best_result(&new, &old, 10, &costs);
        assert!(r.covered[0]);
        assert!(r.covered[1]);
        assert!(!r.covered[2]);
        assert!((r.e_cost - 2.0).abs() < 1e-10);
        assert_eq!(r.match_log.len(), 2);
    }

    #[test]
    fn scenario3_gap_match_within_window() {
        let new = vec![0u32, 1, 2];
        let p0 = Pattern::new_with_gaps(
            0,
            vec![SymbolRef::Atom(0), SymbolRef::Atom(2)],
            vec![crate::model::GapConstraint::up_to(2)],
            0,
        );
        let old = vec![&p0];
        let costs = vec![1.0; 3];
        let r = best_result(&new, &old, 10, &costs);
        assert!(r.covered[0]);
        assert!(!r.covered[1]);
        assert!(r.covered[2]);
        assert!((r.e_cost - 1.0).abs() < 1e-10);
        assert_eq!(r.match_log.len(), 2);
    }

    #[test]
    fn scenario4_gap_rejected_when_skip_exceeds_max() {
        let new = vec![0u32, 1, 2, 3, 4];
        let p0 = Pattern::new_with_gaps(
            0,
            vec![SymbolRef::Atom(0), SymbolRef::Atom(4)],
            vec![crate::model::GapConstraint::up_to(2)],
            0,
        );
        let old = vec![&p0];
        let costs = vec![1.0; 5];
        let r = best_result(&new, &old, 10, &costs);
        assert!(!(r.covered[0] && r.covered[4]));
    }

    #[test]
    fn scenario5_gap_wrong_order() {
        let new = vec![2u32, 1, 0];
        let p0 = Pattern::new_with_gaps(
            0,
            vec![SymbolRef::Atom(0), SymbolRef::Atom(2)],
            vec![crate::model::GapConstraint::up_to(2)],
            0,
        );
        let old = vec![&p0];
        let costs = vec![1.0; 3];
        let r = best_result(&new, &old, 10, &costs);
        assert!(!(r.covered[0] && r.covered[2]));
    }

    #[test]
    fn scenario6_two_non_overlapping_patterns() {
        let new = vec![0u32, 1, 2, 3];
        let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(0), SymbolRef::Atom(1)], 0);
        let p1 = Pattern::new_contiguous(1, vec![SymbolRef::Atom(2), SymbolRef::Atom(3)], 0);
        let old = vec![&p0, &p1];
        let costs = vec![1.0; 4];
        let r = best_result(&new, &old, 20, &costs);
        assert_eq!(r.e_cost, 0.0);
        assert!(r.covered.iter().all(|&c| c));
        assert!(r.match_log.iter().any(|e| e.old_idx == 0));
        assert!(r.match_log.iter().any(|e| e.old_idx == 1));
    }

    #[test]
    fn scenario7_single_symbol_pattern_matches_twice() {
        let new = vec![0u32, 1, 0];
        let p0 = Pattern::new_contiguous(0, vec![SymbolRef::Atom(0)], 0);
        let old = vec![&p0];
        let costs = vec![1.0f64, 2.0];
        let r = best_result(&new, &old, 10, &costs);
        assert!(r.covered[0]);
        assert!(!r.covered[1]);
        assert!(r.covered[2]);
        assert!((r.e_cost - 2.0).abs() < 1e-10);
        assert_eq!(r.match_log.len(), 2);
    }
}
