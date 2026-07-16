use std::collections::HashMap;

use crate::model::Pattern;

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
    old_cursors: HashMap<usize, usize>,
    new_cursors: HashMap<usize, usize>,
    max_covered_new: usize,
    covered_new: Vec<bool>,
    cd: f64,
    log_tail: Option<u32>,
}

impl PartialAlignment {
    fn new(new_len: usize) -> Self {
        Self {
            old_cursors: HashMap::new(),
            new_cursors: HashMap::new(),
            max_covered_new: 0,
            covered_new: vec![false; new_len],
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
        let mut next = self.clone();
        next.covered_new[new_pos] = true;
        next.cd += symbol_cost;
        next.old_cursors.insert(old_idx, old_pos);
        next.new_cursors.insert(old_idx, new_pos);
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

    fn can_extend(&self, old_idx: usize, old_pos: usize, new_pos: usize) -> bool {
        match self.new_cursors.get(&old_idx) {
            None => old_pos == 0 && new_pos >= self.max_covered_new,
            Some(&prev_new) => old_pos > 0 && new_pos == prev_new + 1,
        }
    }

    fn finalize(self, new: &[u32], costs: &[f64], arena: &MatchArena) -> RawAlignment {
        let e_cost: f64 = new
            .iter()
            .enumerate()
            .filter(|&(i, _)| !self.covered_new[i])
            .map(|(_, &id)| costs[id as usize])
            .sum();
        let match_log = arena.collect(self.log_tail);
        RawAlignment {
            match_log,
            covered: self.covered_new,
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
    beam_k: usize,
    costs: &[f64],
) -> Vec<RawAlignment> {
    if new.is_empty() {
        return vec![];
    }

    // For each symbol ID: which (old_idx, old_pos) pairs contain it
    let mut symbol_to_old: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    for (oi, pat) in old.iter().enumerate() {
        for (pos, sym_ref) in pat.symbols.iter().enumerate() {
            if let crate::model::SymbolRef::Atom(sym) = sym_ref {
                symbol_to_old.entry(*sym).or_default().push((oi, pos));
            }
        }
    }

    let mut arena = MatchArena::new();
    let mut candidates = vec![PartialAlignment::new(new.len())];

    for (p, &sym) in new.iter().enumerate() {
        let sym_cost = costs[sym as usize];
        let mut next_candidates = Vec::with_capacity(candidates.len() * 2);

        for candidate in &candidates {
            next_candidates.push(candidate.extend_skip());

            if let Some(matches) = symbol_to_old.get(&sym) {
                for &(oi, q) in matches {
                    if candidate.can_extend(oi, q, p) {
                        next_candidates
                            .push(candidate.extend_match(oi, q, p, sym_cost, &mut arena));
                    }
                }
            }
        }

        next_candidates.sort_by(|a, b| {
            b.cd.partial_cmp(&a.cd)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let a_cov = a.covered_new.iter().filter(|&&c| c).count();
                    let b_cov = b.covered_new.iter().filter(|&&c| c).count();
                    b_cov.cmp(&a_cov)
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
    results.sort_by(|a, b| {
        b.cd.partial_cmp(&a.cd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Pattern, SymbolRef};

    fn contiguous_pattern(id: u32, atoms: &[u32]) -> Pattern {
        Pattern::new_contiguous(
            id,
            atoms.iter().map(|&a| SymbolRef::Atom(a)).collect(),
            0,
        )
    }

    #[test]
    fn test_arena_fork() {
        let mut arena = MatchArena::new();
        let root = arena.push(
            MatchEvent { old_idx: 0, old_pos: 0, new_pos: 0, cost: 1.0 },
            None,
        );
        let branch_a = arena.push(
            MatchEvent { old_idx: 0, old_pos: 1, new_pos: 1, cost: 2.0 },
            Some(root),
        );
        let branch_b = arena.push(
            MatchEvent { old_idx: 1, old_pos: 0, new_pos: 2, cost: 3.0 },
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
        let results = beam_search(&new, &old_refs, 5, &costs);
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
        let results = beam_search(&new, &old_refs, 5, &costs);
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
        let results = beam_search(&new, &old_refs, 10, &costs);
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
        let results = beam_search(&new, &old_refs, 10, &costs);
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
        let results = beam_search(&new, &old_refs, 20, &costs);
        let best = &results[0];
        assert_eq!(best.e_cost, 0.0, "correct order: all symbols must be covered");
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
        let results = beam_search(&new, &old_refs, 20, &costs);
        let best = &results[0];
        assert!(
            best.e_cost > 0.0,
            "interleaved pattern: must not fully cover [A,C,B,D] with [A,B] and [C,D]"
        );
    }

    #[test]
    fn match_log_records_events() {
        // old = [[A, B]], new = [A, B] → match_log has 2 events
        let new = vec![0u32, 1];
        let p0 = contiguous_pattern(0, &[0u32, 1u32]);
        let old_refs = vec![&p0];
        let costs = vec![1.0, 2.0];
        let results = beam_search(&new, &old_refs, 5, &costs);
        let best = &results[0];
        assert_eq!(best.match_log.len(), 2);
        assert_eq!(best.match_log[0].new_pos, 0);
        assert_eq!(best.match_log[1].new_pos, 1);
        assert_eq!(best.match_log[0].old_pos, 0);
        assert_eq!(best.match_log[1].old_pos, 1);
    }
}
