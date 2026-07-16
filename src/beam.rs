use std::collections::HashMap;

use crate::AlignmentType;

/// A finalized beam alignment result.
#[derive(Debug, Clone)]
pub struct BeamAlignment {
    pub covered_new: Vec<bool>,
    pub old_pattern_indices: Vec<usize>,
    pub g: f64,
    pub e: f64,
    pub t: f64,
    pub cd: f64,
    pub alignment_type: AlignmentType,
}

/// Internal partial alignment state during beam search.
#[derive(Debug, Clone)]
struct PartialAlignment {
    old_cursors: HashMap<usize, usize>,
    // Tracks the last matched New position per old pattern — enforces span contiguity.
    new_cursors: HashMap<usize, usize>,
    // Highest New position covered by any Old pattern — enforces inter-pattern ordering.
    max_covered_new: usize,
    covered_new: Vec<bool>,
    covered_cost: f64,
    cd: f64,
}

impl PartialAlignment {
    fn new(new_len: usize) -> Self {
        Self {
            old_cursors: HashMap::new(),
            new_cursors: HashMap::new(),
            max_covered_new: 0,
            covered_new: vec![false; new_len],
            covered_cost: 0.0,
            cd: 0.0,
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
    ) -> Self {
        let mut next = self.clone();
        next.covered_new[new_pos] = true;
        next.covered_cost += symbol_cost;
        // G for existing grammar patterns is 0 (cost was paid at insertion time).
        // CD = covered_cost: bits saved by not encoding matched symbols raw.
        next.old_cursors.insert(old_idx, old_pos);
        next.new_cursors.insert(old_idx, new_pos);
        if new_pos > next.max_covered_new {
            next.max_covered_new = new_pos;
        }
        next.cd = next.covered_cost;
        next
    }

    // `new_pos` is needed to enforce span contiguity for subsequent symbols of the same
    // Old pattern. First symbol of a pattern can start at any New position; thereafter
    // each successive symbol must be at exactly prev_new_pos + 1.
    // Single-symbol patterns (old_pos == prev_old == 0) may re-match at non-contiguous
    // New positions — contiguity only applies when advancing within a multi-symbol pattern.
    // Inter-pattern ordering: first symbol of a NEW pattern (not yet in old_cursors) must
    // start at new_pos >= max_covered_new (patterns consumed left-to-right in New).
    fn can_extend(&self, old_idx: usize, old_pos: usize, new_pos: usize) -> bool {
        match self.old_cursors.get(&old_idx) {
            Some(&prev_old) => {
                if old_pos > prev_old {
                    // Advancing to next symbol in multi-symbol pattern → New must be contiguous.
                    self.new_cursors
                        .get(&old_idx)
                        .map_or(true, |&prev_new| new_pos == prev_new + 1)
                } else {
                    // old_pos == prev_old: single-symbol re-use at a new New position.
                    old_pos >= prev_old
                }
            }
            // First symbol of this Old pattern: must start at old_pos==0 and
            // respect inter-pattern ordering (left-to-right in New).
            None => old_pos == 0 && new_pos >= self.max_covered_new,
        }
    }

    fn finalize(self, new: &[u32], old: &[Vec<u32>], costs: &[f64]) -> BeamAlignment {
        let raw_new_cost: f64 = new.iter().map(|&id| costs[id as usize]).sum();
        let e: f64 = new
            .iter()
            .enumerate()
            .filter(|&(i, _)| !self.covered_new[i])
            .map(|(_, &id)| costs[id as usize])
            .sum();
        // G=0: all patterns in old[] are already in the grammar (cost paid at insertion).
        // T = E (uncovered symbols only). CD = raw - T = covered symbols' cost.
        let g = 0.0;
        let t = g + e;
        let cd = raw_new_cost - t;

        let mut old_pattern_indices: Vec<usize> = self.old_cursors.keys().copied().collect();
        old_pattern_indices.sort_unstable();

        // Determine alignment type
        let new_fully_covered = self.covered_new.iter().all(|&c| c);
        // Check that every symbol in each used Old pattern appears in covered New positions
        let old_all_matched = old_pattern_indices.iter().all(|&oi| {
            let old_pat = &old[oi];
            // Every symbol in this old pattern must appear as a covered match
            // We need to check that the old pattern is fully "consumed"
            // Since we track cursors monotonically, check if cursor reached the last position
            // Actually: a single-symbol old pattern matched at pos 0 means cursor=0, len=1 → fully matched
            // For multi-symbol: we need all symbols to have been matched
            // The beam search only matches one symbol per new position, so for full coverage
            // of an old pattern, we need len(old_pat) symbols matched from it.
            // We can count how many new positions are covered by this old pattern.
            // But we don't track per-old-pattern match count in this simplified version.
            // Alternative: check if old pattern length == number of new positions matched from it.
            // We don't have that info directly. Use a simpler heuristic:
            // For FullA, all old pattern symbols must appear somewhere in covered new positions.
            // This is an approximation — check that every symbol in old[oi] appears in covered new.
            let covered_syms: Vec<u32> = new
                .iter()
                .enumerate()
                .filter(|&(i, _)| self.covered_new[i])
                .map(|(_, &id)| id)
                .collect();
            old_pat.iter().all(|sym| covered_syms.contains(sym))
        });

        let alignment_type = if new_fully_covered && old_all_matched {
            AlignmentType::FullA
        } else if !new_fully_covered && old_all_matched {
            AlignmentType::FullB
        } else {
            AlignmentType::Partial
        };

        BeamAlignment {
            covered_new: self.covered_new,
            old_pattern_indices,
            g,
            e,
            t,
            cd,
            alignment_type,
        }
    }
}

impl BeamAlignment {
    /// Returns indices of Old patterns that contributed at least one matched symbol.
    pub fn matched_old_pattern_ids(&self) -> Vec<usize> {
        self.old_pattern_indices.clone()
    }
}

/// Staged beam search: find top-K alignments between a New pattern and Old patterns.
///
/// - `new`: symbol IDs of the new pattern
/// - `old`: each old pattern as a vec of symbol IDs
/// - `beam_k`: beam width (keep top-K partial alignments at each step)
/// - `costs`: cost table indexed by symbol ID
///
/// Returns top-K complete alignments sorted by CD descending.
pub fn beam_search(
    new: &[u32],
    old: &[Vec<u32>],
    beam_k: usize,
    costs: &[f64],
) -> Vec<BeamAlignment> {
    if new.is_empty() {
        return vec![];
    }

    // Precompute: for each symbol ID, which (old_idx, position) pairs contain it
    let mut symbol_to_old: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    for (oi, pat) in old.iter().enumerate() {
        for (pos, &sym) in pat.iter().enumerate() {
            symbol_to_old.entry(sym).or_default().push((oi, pos));
        }
    }

    let mut candidates = vec![PartialAlignment::new(new.len())];

    for (p, &sym) in new.iter().enumerate() {
        let sym_cost = costs[sym as usize];
        let mut next_candidates = Vec::with_capacity(candidates.len() * 2);

        for candidate in &candidates {
            // Option A: skip this position
            next_candidates.push(candidate.extend_skip());

            // Option B: match against old patterns
            if let Some(matches) = symbol_to_old.get(&sym) {
                for &(oi, q) in matches {
                    if candidate.can_extend(oi, q, p) {
                        next_candidates.push(candidate.extend_match(oi, q, p, sym_cost));
                    }
                }
            }
        }

        // Prune to beam_k: sort by CD descending, break ties by coverage count, then lexicographically
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

    // Finalize and sort by CD descending
    let mut results: Vec<BeamAlignment> = candidates
        .into_iter()
        .map(|c| c.finalize(new, old, costs))
        .collect();
    results.sort_by(|a, b| {
        b.cd.partial_cmp(&a.cd)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.old_pattern_indices.cmp(&b.old_pattern_indices))
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_empty_old() {
        let new = vec![0u32, 1, 2];
        let old: Vec<Vec<u32>> = vec![];
        let costs = vec![1.0, 2.0, 3.0];
        let results = beam_search(&new, &old, 5, &costs);
        assert_eq!(results.len(), 1);
        assert!(results[0].covered_new.iter().all(|&c| !c));
        assert_eq!(results[0].cd, 0.0);
    }

    #[test]
    fn test_beam_single_match() {
        let new = vec![0u32, 1];
        let old = vec![vec![0u32]];
        let costs = vec![2.0, 3.0];
        let results = beam_search(&new, &old, 5, &costs);
        assert!(!results.is_empty());
        let best = &results[0];
        assert!(best.covered_new[0]);
    }

    #[test]
    fn span_contiguity_non_contiguous_gap_not_covered() {
        // old = [[A, B]], new = [A, X, B]
        // B is at new[2], but A matched at new[0] → next must be new[1], not new[2].
        // B at new[2] must NOT be covered.
        // IDs: A=0, X=1, B=2
        let new = vec![0u32, 1, 2];
        let old = vec![vec![0u32, 2u32]]; // [A, B]
        let costs = vec![1.0, 1.0, 1.0];
        let results = beam_search(&new, &old, 10, &costs);
        let best = &results[0];
        // A at new[0] may be covered (first symbol, any start OK)
        // B at new[2] must NOT be covered (gap at new[1])
        assert!(
            !best.covered_new[2],
            "B at new[2] must not be covered: gap between A(new[0]) and B(new[2])"
        );
    }

    #[test]
    fn span_contiguity_contiguous_pair_both_covered() {
        // old = [[A, B]], new = [A, B, C]
        // A at new[0], B at new[1] → contiguous → both covered. C uncovered.
        // IDs: A=0, B=1, C=2
        let new = vec![0u32, 1, 2];
        let old = vec![vec![0u32, 1u32]]; // [A, B]
        let costs = vec![1.0, 1.0, 1.0];
        let results = beam_search(&new, &old, 10, &costs);
        let best = &results[0];
        assert!(best.covered_new[0], "A at new[0] should be covered");
        assert!(best.covered_new[1], "B at new[1] should be covered");
        assert!(!best.covered_new[2], "C at new[2] should not be covered");
    }

    #[test]
    fn inter_pattern_order_correct_order_fully_covered() {
        // old = [[A, B], [C, D]], new = [A, B, C, D]
        // Patterns in correct New-order → E = 0.
        // IDs: A=0, B=1, C=2, D=3
        let new = vec![0u32, 1, 2, 3];
        let old = vec![vec![0u32, 1u32], vec![2u32, 3u32]];
        let costs = vec![1.0, 1.0, 1.0, 1.0];
        let results = beam_search(&new, &old, 20, &costs);
        let best = &results[0];
        assert_eq!(best.e, 0.0, "correct order: all symbols must be covered");
        assert!(best.covered_new.iter().all(|&c| c));
    }

    #[test]
    fn inter_pattern_order_interleaved_rejected() {
        // old = [[A, B], [C, D]], new = [A, C, B, D]
        // [A,B] can start at new[0] (A). Then [C,D] would need to start >= 0 — C is at new[1] >= 0 OK.
        // But then [A,B] needs B at new[2] which is contiguous with A at new[0]? No: new[2] != new[0]+1.
        // Span contiguity (Issue #3) blocks B at new[2] for [A,B] that started at new[0].
        // [C,D] can match C at new[1], then D must be at new[2] — but new[2]=B not D → blocked.
        // Net result: at most one symbol covered per pattern → E > 0.
        // IDs: A=0, B=1, C=2, D=3; new order: A C B D
        let new = vec![0u32, 2, 1, 3];
        let old = vec![vec![0u32, 1u32], vec![2u32, 3u32]];
        let costs = vec![1.0, 1.0, 1.0, 1.0];
        let results = beam_search(&new, &old, 20, &costs);
        let best = &results[0];
        assert!(
            best.e > 0.0,
            "interleaved pattern: must not fully cover [A,C,B,D] with [A,B] and [C,D]"
        );
    }
}
