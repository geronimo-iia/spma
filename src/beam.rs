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
    covered_new: Vec<bool>,
    covered_cost: f64,
    cd: f64,
}

impl PartialAlignment {
    fn new(new_len: usize) -> Self {
        Self {
            old_cursors: HashMap::new(),
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
        next.cd = next.covered_cost;
        next
    }

    fn can_extend(&self, old_idx: usize, old_pos: usize) -> bool {
        match self.old_cursors.get(&old_idx) {
            Some(&prev) => old_pos >= prev,
            None => true,
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

        let old_pattern_indices: Vec<usize> = self.old_cursors.keys().copied().collect();

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
                    if candidate.can_extend(oi, q) {
                        next_candidates.push(candidate.extend_match(oi, q, p, sym_cost));
                    }
                }
            }
        }

        // Prune to beam_k: sort by CD descending, break ties by coverage count descending
        next_candidates.sort_by(|a, b| {
            b.cd.partial_cmp(&a.cd)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let a_cov = a.covered_new.iter().filter(|&&c| c).count();
                    let b_cov = b.covered_new.iter().filter(|&&c| c).count();
                    b_cov.cmp(&a_cov)
                })
        });
        next_candidates.truncate(beam_k);
        candidates = next_candidates;
    }

    // Finalize and sort by CD descending
    let mut results: Vec<BeamAlignment> = candidates
        .into_iter()
        .map(|c| c.finalize(new, old, costs))
        .collect();
    results.sort_by(|a, b| b.cd.partial_cmp(&a.cd).unwrap_or(std::cmp::Ordering::Equal));
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
}
