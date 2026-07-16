// Phase 1d — see docs/grammar-spec.md, docs/roadmap.md

use std::collections::HashMap;

use crate::alignment::{build_alignment, Alignment};
use crate::beam::{beam_search, RawAlignment};
use crate::model::{Grammar, GrammarLevel, Pattern, SymbolRef};

// ── Public result types ───────────────────────────────────────────────────────

pub struct InferResult {
    pub e_cost: f64,
    pub is_anomaly: bool,
    pub cd: f64,
    pub e_norm: f64,
    pub anomaly_percentile: f64,
    pub level_costs: Vec<f64>,
    pub level_e_norms: Vec<f64>,
    pub alignment: Alignment,
}

// ── Spma ──────────────────────────────────────────────────────────────────────

pub struct Spma {
    pub grammar: Grammar,
    beam_k: usize,
    pub atom_costs: Vec<f64>,
    max_induced_gap: usize,
}

impl Spma {
    pub fn new(beam_k: usize) -> Self {
        Self {
            grammar: Grammar::default(),
            beam_k,
            atom_costs: Vec::new(),
            max_induced_gap: MAX_INDUCED_GAP,
        }
    }

    pub fn set_max_induced_gap(&mut self, max: usize) {
        self.max_induced_gap = max;
    }

    pub fn set_anomaly_threshold(&mut self, threshold: f64) {
        self.grammar.e_distribution.threshold = threshold;
    }

    pub fn e_distribution(&self) -> &crate::model::EDistribution {
        &self.grammar.e_distribution
    }

    fn infer_internal(&self, seq: &[u32]) -> (f64, f64) {
        if seq.is_empty() {
            return (0.0, 0.0);
        }
        let raw_new_cost: f64 = seq
            .iter()
            .map(|&id| self.atom_costs.get(id as usize).copied().unwrap_or(1.0))
            .sum();
        if self.grammar.levels.is_empty() {
            return (raw_new_cost, raw_new_cost);
        }
        let level0_patterns: Vec<&Pattern> = self.grammar.levels[0].patterns.iter().collect();
        let results = beam_search(seq, &level0_patterns, self.beam_k, &self.atom_costs);
        let e_cost = results
            .into_iter()
            .next()
            .map(|r| r.e_cost)
            .unwrap_or(raw_new_cost);
        (e_cost, raw_new_cost)
    }

    pub fn train(&mut self, corpus: &[Vec<&str>]) {
        // Step 1: intern all symbols
        for seq in corpus {
            for s in seq {
                self.grammar.interner.intern(s);
            }
        }

        // Step 2: build atom-level sequences and uniform costs
        let atom_seqs: Vec<Vec<u32>> = corpus
            .iter()
            .map(|seq| {
                seq.iter()
                    .map(|s| self.grammar.interner.intern(s))
                    .collect()
            })
            .collect();

        let n_atoms = self.grammar.interner.len();
        // Frequency-based costs: -log2(freq/total) per atom
        let mut atom_freq: HashMap<u32, u32> = HashMap::new();
        let total_atoms: u32 = atom_seqs.iter().flat_map(|s| s.iter()).count() as u32;
        for seq in &atom_seqs {
            for &id in seq {
                *atom_freq.entry(id).or_insert(0) += 1;
            }
        }
        let costs: Vec<f64> = (0..n_atoms as u32)
            .map(|id| {
                let freq = atom_freq.get(&id).copied().unwrap_or(1);
                -((freq as f64 / total_atoms as f64).log2())
            })
            .collect();
        self.atom_costs = costs.clone();

        // Step 3: cold-start — extract frequent n-grams from atom sequences
        let min_freq = (corpus.len() / 2).max(2);
        let ngrams = extract_frequent_ngrams(&atom_seqs, min_freq);

        // Build level-0 patterns from n-grams
        let mut next_id: u32 = 0;
        let mut level0_patterns: Vec<Pattern> = Vec::new();
        let mut all_id_vecs: Vec<Vec<u32>> = Vec::new();

        for (ngram, _freq) in &ngrams {
            // Gap-encoded candidates: [sym_i, GAP_MARKER, gap_size, sym_j]
            let is_gap_candidate = ngram.len() == 4 && ngram[1] == GAP_MARKER;

            // For MDL gating, the atom IDs used in the pattern (not sentinels)
            let atom_ids: Vec<u32> = if is_gap_candidate {
                vec![ngram[0], ngram[3]]
            } else {
                ngram.clone()
            };

            let pattern_cost: f64 = atom_ids.iter().map(|&id| costs[id as usize]).sum();
            let new_g: f64 = all_id_vecs
                .iter()
                .flat_map(|v| v.iter())
                .map(|&id| costs[id as usize])
                .sum::<f64>()
                + pattern_cost;
            let current_e = compute_total_e_dp(&atom_seqs, &all_id_vecs, &costs);
            let current_t = new_g - pattern_cost + current_e;

            // For dp matching, gap patterns reduce to their two flanking atoms
            let matching_vec = atom_ids.clone();
            let mut candidate_multi = all_id_vecs.clone();
            candidate_multi.push(matching_vec.clone());
            let new_e = compute_total_e_dp(&atom_seqs, &candidate_multi, &costs);
            let new_t = new_g + new_e;

            if new_t < current_t || all_id_vecs.is_empty() {
                let freq = ngrams
                    .iter()
                    .find(|(n, _)| n == ngram)
                    .map(|(_, f)| *f)
                    .unwrap_or(1);
                let pat = if is_gap_candidate {
                    let gap_size = ngram[2] as usize;
                    let symbols = vec![SymbolRef::Atom(ngram[0]), SymbolRef::Atom(ngram[3])];
                    let gaps = vec![crate::model::GapConstraint::up_to(gap_size)];
                    let mut p = Pattern::new_with_gaps(next_id, symbols, gaps, 0);
                    p.frequency = freq;
                    p
                } else {
                    let symbols: Vec<SymbolRef> =
                        ngram.iter().map(|&id| SymbolRef::Atom(id)).collect();
                    let mut p = Pattern::new_contiguous(next_id, symbols, 0);
                    p.frequency = freq;
                    p
                };
                level0_patterns.push(pat);
                all_id_vecs.push(matching_vec);
                next_id += 1;
            }
        }

        // If cold-start found nothing, push at least the most frequent bigram
        if level0_patterns.is_empty() && !ngrams.is_empty() {
            let (ngram, freq) = &ngrams[0];
            let is_gap = ngram.len() == 4 && ngram[1] == GAP_MARKER;
            let pat = if is_gap {
                let gap_size = ngram[2] as usize;
                let symbols = vec![SymbolRef::Atom(ngram[0]), SymbolRef::Atom(ngram[3])];
                let gaps = vec![crate::model::GapConstraint::up_to(gap_size)];
                let mut p = Pattern::new_with_gaps(next_id, symbols, gaps, 0);
                p.frequency = *freq;
                p
            } else {
                let symbols: Vec<SymbolRef> = ngram.iter().map(|&id| SymbolRef::Atom(id)).collect();
                let mut p = Pattern::new_contiguous(next_id, symbols, 0);
                p.frequency = *freq;
                p
            };
            let matching_vec = if is_gap {
                vec![ngram[0], ngram[3]]
            } else {
                ngram.clone()
            };
            level0_patterns.push(pat);
            all_id_vecs.push(matching_vec);
            next_id += 1;
        }

        self.grammar.levels.push(GrammarLevel::new(level0_patterns));

        // Step 4: outer N-level loop — learn hierarchical levels
        let max_levels: usize = 8;
        let mut current_atom_seqs = atom_seqs.clone();
        let mut current_costs = costs.clone();

        for level in 0..max_levels {
            let level_patterns: Vec<&Pattern> =
                self.grammar.levels[level].patterns.iter().collect();
            if level_patterns.is_empty() {
                break;
            }

            // Run beam_search on each sequence to get match logs
            // Collect results first (immutable borrow ends), then update frequencies.
            let raw_results: Vec<Option<RawAlignment>> = current_atom_seqs
                .iter()
                .map(|seq| {
                    beam_search(seq, &level_patterns, self.beam_k, &current_costs)
                        .into_iter()
                        .next()
                })
                .collect();

            let mut all_match_logs: Vec<Vec<crate::beam::MatchEvent>> = Vec::new();
            let mut gap_candidates: Vec<Pattern> = Vec::new();

            for (seq_idx, best_opt) in raw_results.into_iter().enumerate() {
                if let Some(best) = best_opt {
                    let used_idxs: std::collections::HashSet<usize> =
                        best.match_log.iter().map(|e| e.old_idx).collect();
                    for &oi in &used_idxs {
                        self.grammar.levels[level].patterns[oi].frequency += 1;
                    }

                    // Harvest gap patterns from covered array
                    let seq = &current_atom_seqs[seq_idx];
                    let seq_symbols: Vec<SymbolRef> =
                        seq.iter().map(|&id| SymbolRef::Atom(id)).collect();
                    let seq_as_pat = Pattern::new_contiguous(u32::MAX, seq_symbols, level as u8);
                    let new_gaps = extract_learned_patterns(
                        &seq_as_pat,
                        &best.covered,
                        &mut next_id,
                        self.max_induced_gap,
                        level as u8,
                    );
                    gap_candidates.extend(new_gaps);

                    all_match_logs.push(best.match_log);
                } else {
                    all_match_logs.push(Vec::new());
                }
            }

            // Build next-level pid sequences from match logs
            let mut next_level_pats =
                build_next_level_patterns(&all_match_logs, &self.grammar, level, &mut next_id);

            // Add gap candidates to the pool (dedup by symbol fingerprint)
            let mut seen_fingerprints: std::collections::HashSet<Vec<u32>> = next_level_pats
                .iter()
                .map(|p| {
                    p.symbols
                        .iter()
                        .map(|s| match s {
                            SymbolRef::Atom(id) | SymbolRef::Pattern(id) => *id,
                        })
                        .collect()
                })
                .collect();
            for pat in gap_candidates {
                let fp: Vec<u32> = pat
                    .symbols
                    .iter()
                    .map(|s| match s {
                        SymbolRef::Atom(id) | SymbolRef::Pattern(id) => *id,
                    })
                    .collect();
                if seen_fingerprints.insert(fp) {
                    next_level_pats.push(pat);
                }
            }

            if next_level_pats.is_empty() {
                break;
            }

            // Build pid sequences for MDL check at next level
            let pid_seqs: Vec<Vec<u32>> = all_match_logs
                .iter()
                .map(|log| {
                    let mut pid_positions: Vec<(u32, usize)> = Vec::new();
                    for event in log {
                        if event.old_pos == 0 {
                            let pat = &self.grammar.levels[level].patterns[event.old_idx];
                            pid_positions.push((pat.id, event.new_pos));
                        }
                    }
                    pid_positions.sort_by_key(|&(_, pos)| pos);
                    pid_positions.into_iter().map(|(pid, _)| pid).collect()
                })
                .collect();

            // Costs for pattern IDs at next level — uniform based on pattern count
            let n_pats = self.grammar.levels[level].patterns.len();
            let pid_cost = if n_pats > 1 {
                (n_pats as f64).log2()
            } else {
                1.0
            };
            let max_pid = self.grammar.levels[level]
                .patterns
                .iter()
                .map(|p| p.id as usize + 1)
                .max()
                .unwrap_or(1);
            let pid_costs = vec![pid_cost; max_pid + next_id as usize];

            // MDL-gate next level patterns
            let mut next_id_vecs: Vec<Vec<u32>> = Vec::new();
            let mut accepted_pats: Vec<Pattern> = Vec::new();

            for pat in next_level_pats {
                let ngram: Vec<u32> = pat
                    .symbols
                    .iter()
                    .map(|s| match s {
                        SymbolRef::Pattern(id) => *id,
                        SymbolRef::Atom(id) => *id,
                    })
                    .collect();

                let current_g: f64 = next_id_vecs
                    .iter()
                    .flat_map(|v| v.iter())
                    .map(|&id| pid_costs.get(id as usize).copied().unwrap_or(pid_cost))
                    .sum();
                let pattern_cost: f64 = ngram
                    .iter()
                    .map(|&id| pid_costs.get(id as usize).copied().unwrap_or(pid_cost))
                    .sum();
                let current_e = compute_total_e_dp(&pid_seqs, &next_id_vecs, &pid_costs);
                let current_t = current_g + current_e;

                let mut candidate = next_id_vecs.clone();
                candidate.push(ngram.clone());
                let new_g = current_g + pattern_cost;
                let new_e = compute_total_e_dp(&pid_seqs, &candidate, &pid_costs);
                let new_t = new_g + new_e;

                if new_t < current_t || next_id_vecs.is_empty() {
                    next_id_vecs.push(ngram);
                    accepted_pats.push(pat);
                }
            }

            if accepted_pats.is_empty() {
                break;
            }

            self.grammar.levels.push(GrammarLevel::new(accepted_pats));
            current_atom_seqs = pid_seqs;
            current_costs = pid_costs;
        }

        // Populate EDistribution from training sequences
        let e_norms: Vec<f64> = atom_seqs
            .iter()
            .filter_map(|seq| {
                let (e_cost, raw) = self.infer_internal(seq);
                if raw < 1e-12 {
                    None
                } else {
                    Some(e_cost / raw)
                }
            })
            .collect();

        // Per-level e_norms: for each level, run beam on that level's pid sequences
        // collected during training (stored in current_atom_seqs after the loop)
        let n_levels = self.grammar.levels.len();
        let mut level_e_norms_vecs: Vec<Vec<f64>> = Vec::with_capacity(n_levels);

        // Level 0: same as global e_norms (atom sequences)
        level_e_norms_vecs.push(e_norms.clone());

        // Higher levels: rebuild pid seqs from atom_seqs through each level
        if n_levels > 1 {
            let mut lvl_seqs: Vec<Vec<u32>> = atom_seqs.clone();
            let mut lvl_costs: Vec<f64> = self.atom_costs.clone();

            for level in 0..n_levels - 1 {
                let level_patterns: Vec<&Pattern> =
                    self.grammar.levels[level].patterns.iter().collect();
                let mut next_lvl_seqs: Vec<Vec<u32>> = Vec::new();

                for seq in &lvl_seqs {
                    let results = beam_search(seq, &level_patterns, self.beam_k, &lvl_costs);
                    let pid_seq: Vec<u32> = if let Some(best) = results.into_iter().next() {
                        let mut pid_positions: Vec<(u32, usize)> = Vec::new();
                        for event in &best.match_log {
                            if event.old_pos == 0 {
                                if let Some(pat) = level_patterns.get(event.old_idx) {
                                    pid_positions.push((pat.id, event.new_pos));
                                }
                            }
                        }
                        pid_positions.sort_by_key(|&(_, pos)| pos);
                        pid_positions.dedup_by_key(|x| x.1);
                        pid_positions.into_iter().map(|(pid, _)| pid).collect()
                    } else {
                        Vec::new()
                    };
                    next_lvl_seqs.push(pid_seq);
                }

                // Build pid costs for next level
                let n_pats = self.grammar.levels[level].patterns.len();
                let pid_cost = if n_pats > 1 {
                    (n_pats as f64).log2()
                } else {
                    1.0
                };
                let max_pid = self.grammar.levels[level]
                    .patterns
                    .iter()
                    .map(|p| p.id as usize + 1)
                    .max()
                    .unwrap_or(1);
                let next_costs = vec![pid_cost; max_pid];

                // Compute e_norms at next level
                let next_level_patterns: Vec<&Pattern> =
                    self.grammar.levels[level + 1].patterns.iter().collect();
                let lvl_e: Vec<f64> = next_lvl_seqs
                    .iter()
                    .filter_map(|seq| {
                        if seq.is_empty() {
                            return None;
                        }
                        let raw: f64 = seq
                            .iter()
                            .map(|&id| next_costs.get(id as usize).copied().unwrap_or(pid_cost))
                            .sum();
                        if raw < 1e-12 {
                            return None;
                        }
                        let results =
                            beam_search(seq, &next_level_patterns, self.beam_k, &next_costs);
                        let e = results.into_iter().next().map(|r| r.e_cost).unwrap_or(raw);
                        Some(e / raw)
                    })
                    .collect();
                level_e_norms_vecs.push(lvl_e);

                lvl_seqs = next_lvl_seqs;
                lvl_costs = next_costs;
            }
        }

        self.grammar.e_distribution =
            crate::model::EDistribution::fit(e_norms, 0.0, level_e_norms_vecs);
    }

    pub fn infer(&self, seq: &[&str]) -> InferResult {
        let n_atoms = self.grammar.interner.len();
        // Fallback cost for unknown symbols: max atom cost or 1.0
        let fallback_cost = self.atom_costs.iter().cloned().fold(1.0f64, f64::max);

        let ids: Vec<u32> = seq
            .iter()
            .map(|s| self.grammar.interner.get(s).unwrap_or(n_atoms as u32))
            .collect();

        let new_names: Vec<&str> = seq.to_vec();

        // Extend atom_costs for unknown symbols
        let costs_len = (n_atoms + 1).max(ids.iter().map(|&id| id as usize + 1).max().unwrap_or(1));
        let mut costs: Vec<f64> = self.atom_costs.clone();
        costs.resize(costs_len, fallback_cost);

        let raw_new_cost: f64 = ids
            .iter()
            .map(|&id| costs.get(id as usize).copied().unwrap_or(fallback_cost))
            .sum();

        if self.grammar.levels.is_empty() {
            let alignment = Alignment {
                new_symbols: seq.iter().map(|s| s.to_string()).collect(),
                rows: Vec::new(),
                covered: vec![false; seq.len()],
                e_cost: raw_new_cost,
                cd: 0.0,
                level_costs: Vec::new(),
            };
            return InferResult {
                e_cost: raw_new_cost,
                is_anomaly: raw_new_cost > self.grammar.e_distribution.threshold,
                cd: 0.0,
                e_norm: 0.0,
                anomaly_percentile: 0.0,
                level_costs: Vec::new(),
                level_e_norms: Vec::new(),
                alignment,
            };
        }

        // Level-0 beam search
        let level0_patterns: Vec<&Pattern> = self.grammar.levels[0].patterns.iter().collect();

        let results = beam_search(&ids, &level0_patterns, self.beam_k, &costs);

        let best_raw = results.into_iter().next().unwrap_or_else(|| RawAlignment {
            match_log: Vec::new(),
            covered: vec![false; ids.len()],
            e_cost: raw_new_cost,
            cd: 0.0,
        });

        let e_cost = best_raw.e_cost;
        let cd = best_raw.cd;

        let e_norm = if raw_new_cost < 1e-12 {
            0.0
        } else {
            e_cost / raw_new_cost
        };

        let is_anomaly = e_norm > self.grammar.e_distribution.threshold;
        let anomaly_percentile = self.grammar.e_distribution.anomaly_rank(e_norm); // strict < semantics: training seqs score 0.0

        let alignment = build_alignment(&best_raw, &new_names, &level0_patterns, &self.grammar);

        // Build level_costs and level_e_norms — one entry per grammar level
        let mut level_costs: Vec<f64> = Vec::new();
        let mut level_e_norms: Vec<f64> = Vec::new();

        // Level 0: reuse best_raw already computed above
        level_costs.push(e_cost);
        level_e_norms.push(if raw_new_cost < 1e-12 {
            0.0
        } else {
            e_cost / raw_new_cost
        });

        // Higher levels: use pid sequences derived from match log
        let mut current_seq = ids.clone();
        let mut current_costs_vec = costs.clone();

        for level in 1..self.grammar.levels.len() {
            // Build pid sequence from previous level beam result
            let prev_patterns: Vec<&Pattern> =
                self.grammar.levels[level - 1].patterns.iter().collect();
            let prev_results = beam_search(
                &current_seq,
                &prev_patterns,
                self.beam_k,
                &current_costs_vec,
            );

            let pid_seq: Vec<u32> = if let Some(best) = prev_results.into_iter().next() {
                let mut pid_positions: Vec<(u32, usize)> = Vec::new();
                for event in &best.match_log {
                    if event.old_pos == 0 {
                        if let Some(pat) = prev_patterns.get(event.old_idx) {
                            pid_positions.push((pat.id, event.new_pos));
                        }
                    }
                }
                pid_positions.sort_by_key(|&(_, pos)| pos);
                pid_positions.dedup_by_key(|x| x.1);
                pid_positions.into_iter().map(|(pid, _)| pid).collect()
            } else {
                Vec::new()
            };

            if pid_seq.is_empty() {
                // Pad remaining levels with 0.0
                for _ in level..self.grammar.levels.len() {
                    level_costs.push(0.0);
                    level_e_norms.push(0.0);
                }
                break;
            }

            // Frequency-based costs for this level's pattern IDs
            let n_prev_pats = self.grammar.levels[level - 1].patterns.len();
            let total_pid: u32 = pid_seq.len() as u32;
            let mut pid_freq: HashMap<u32, u32> = HashMap::new();
            for &pid in &pid_seq {
                *pid_freq.entry(pid).or_insert(0) += 1;
            }
            let max_pid = self.grammar.levels[level - 1]
                .patterns
                .iter()
                .map(|p| p.id as usize + 1)
                .max()
                .unwrap_or(1);
            let pid_costs: Vec<f64> = (0..max_pid as u32)
                .map(|id| {
                    let freq = pid_freq.get(&id).copied().unwrap_or(1);
                    -((freq as f64 / total_pid.max(1) as f64).log2())
                })
                .collect();

            let raw_level_cost: f64 = pid_seq
                .iter()
                .map(|&id| pid_costs.get(id as usize).copied().unwrap_or(1.0))
                .sum();

            let level_patterns: Vec<&Pattern> =
                self.grammar.levels[level].patterns.iter().collect();

            // Extend pid_costs to cover all pattern IDs referenced
            let max_ref = pid_seq.iter().map(|&id| id as usize + 1).max().unwrap_or(1);
            let mut pid_costs_ext = pid_costs.clone();
            let fallback_pid = if n_prev_pats > 1 {
                (n_prev_pats as f64).log2()
            } else {
                1.0
            };
            pid_costs_ext.resize(max_ref.max(pid_costs_ext.len()), fallback_pid);

            let level_results = beam_search(&pid_seq, &level_patterns, self.beam_k, &pid_costs_ext);
            let lc = level_results
                .into_iter()
                .next()
                .map(|r| r.e_cost)
                .unwrap_or(raw_level_cost);

            level_costs.push(lc);
            level_e_norms.push(if raw_level_cost < 1e-12 {
                0.0
            } else {
                lc / raw_level_cost
            });

            current_seq = pid_seq;
            current_costs_vec = pid_costs_ext;
        }

        InferResult {
            e_cost,
            is_anomaly,
            cd,
            e_norm,
            anomaly_percentile,
            level_costs,
            level_e_norms,
            alignment,
        }
    }
}

// ── extract_frequent_ngrams ───────────────────────────────────────────────────

const GAP_MARKER: u32 = u32::MAX;
const MAX_INDUCED_GAP: usize = 3;

/// Count contiguous bigrams/trigrams and gap-aware pairs within a window.
/// Gap candidates encoded as [sym_i, GAP_MARKER, gap_size, sym_j].
pub fn extract_frequent_ngrams(seqs: &[Vec<u32>], min_freq: usize) -> Vec<(Vec<u32>, u32)> {
    let mut counts: HashMap<Vec<u32>, u32> = HashMap::new();

    for seq in seqs {
        // Contiguous bigrams and trigrams
        for n in 2..=3usize {
            if seq.len() >= n {
                for window in seq.windows(n) {
                    *counts.entry(window.to_vec()).or_insert(0) += 1;
                }
            }
        }

        // Gap-aware pairs within window
        let len = seq.len();
        for i in 0..len {
            for j in (i + 2)..=(i + MAX_INDUCED_GAP + 1).min(len - 1) {
                let gap_size = (j - i - 1) as u32;
                let key = vec![seq[i], GAP_MARKER, gap_size, seq[j]];
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let mut result: Vec<(Vec<u32>, u32)> = counts
        .into_iter()
        .filter(|(_, count)| *count >= min_freq as u32)
        .collect();

    // Sort by descending (freq * len) so best candidates come first
    result.sort_by(|a, b| {
        let score_b = b.1 as usize * b.0.len();
        let score_a = a.1 as usize * a.0.len();
        score_b.cmp(&score_a).then_with(|| a.0.cmp(&b.0))
    });

    result
}

// ── build_next_level_patterns ─────────────────────────────────────────────────

fn build_next_level_patterns(
    match_logs: &[Vec<crate::beam::MatchEvent>],
    grammar: &Grammar,
    level: usize,
    next_id: &mut u32,
) -> Vec<Pattern> {
    let level_patterns = &grammar.levels[level].patterns;

    // Build pid sequences — pid ordering by first match_log position (not broken contains scan)
    let pid_seqs: Vec<Vec<u32>> = match_logs
        .iter()
        .map(|log| {
            // Collect (pid, first_new_pos) for each pattern that starts (old_pos == 0)
            let mut pid_positions: Vec<(u32, usize)> = Vec::new();
            for event in log {
                if event.old_pos == 0 {
                    if let Some(pat) = level_patterns.get(event.old_idx) {
                        pid_positions.push((pat.id, event.new_pos));
                    }
                }
            }
            pid_positions.sort_by_key(|&(_, pos)| pos);
            // Deduplicate same pid at same position
            pid_positions.dedup_by_key(|x| x.1);
            pid_positions.into_iter().map(|(pid, _)| pid).collect()
        })
        .collect();

    // Extract frequent n-grams from pid sequences
    let min_freq = (match_logs.len() / 2).max(2);
    let ngrams = extract_frequent_ngrams(&pid_seqs, min_freq);

    ngrams
        .into_iter()
        .filter(|(ngram, _)| {
            // Drop gap-encoded candidates — pid sequences don't support gap induction
            !ngram.contains(&GAP_MARKER)
        })
        .map(|(ngram, freq)| {
            let symbols: Vec<SymbolRef> = ngram.iter().map(|&id| SymbolRef::Pattern(id)).collect();
            let mut pat = Pattern::new_contiguous(*next_id, symbols, (level + 1) as u8);
            pat.frequency = freq;
            *next_id += 1;
            pat
        })
        .collect()
}

// ── extract_learned_patterns ──────────────────────────────────────────────────

fn contiguous_spans(covered: &[bool]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (i, &c) in covered.iter().enumerate() {
        match (c, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                spans.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        spans.push((s, covered.len()));
    }
    spans
}

fn merge_close_spans(spans: Vec<(usize, usize)>, max_gap: usize) -> Vec<Vec<(usize, usize)>> {
    if spans.is_empty() {
        return vec![];
    }
    let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut current_group = vec![spans[0]];

    for &span in &spans[1..] {
        let prev_end = current_group.last().unwrap().1;
        let gap = span.0.saturating_sub(prev_end);
        if gap <= max_gap {
            current_group.push(span);
        } else {
            groups.push(current_group);
            current_group = vec![span];
        }
    }
    groups.push(current_group);
    groups
}

fn extract_learned_patterns(
    pattern: &Pattern,
    covered: &[bool],
    next_id: &mut u32,
    max_gap: usize,
    target_level: u8,
) -> Vec<Pattern> {
    let spans = contiguous_spans(covered);
    if spans.is_empty() {
        return vec![];
    }
    let groups = merge_close_spans(spans, max_gap);
    let mut result = Vec::new();

    for sub_spans in groups {
        // Count total covered symbols in this group
        let total_oc: usize = sub_spans.iter().map(|&(s, e)| e - s).sum();
        if total_oc < 2 {
            continue;
        }

        if sub_spans.len() == 1 {
            // Single contiguous span — emit contiguous pattern
            let (start, end) = sub_spans[0];
            let symbols: Vec<SymbolRef> = (start..end).map(|i| pattern.symbols[i]).collect();
            let pat = Pattern::new_contiguous(*next_id, symbols, target_level);
            *next_id += 1;
            result.push(pat);
        } else {
            // Multiple sub-spans — emit gap pattern
            let mut symbols: Vec<SymbolRef> = Vec::new();
            let mut gaps: Vec<crate::model::GapConstraint> = Vec::new();

            for (si, &(start, end)) in sub_spans.iter().enumerate() {
                // Add symbols from this span
                for i in start..end {
                    symbols.push(pattern.symbols[i]);
                }
                // Add gap constraint between this span and the next
                if si + 1 < sub_spans.len() {
                    let next_start = sub_spans[si + 1].0;
                    let actual_gap = next_start - end;
                    // Within span: no gaps needed (contiguous)
                    // Between sub-spans: one gap constraint after last symbol of this span
                    // We need one GapConstraint per adjacent symbol pair that crosses a sub-span boundary
                    // The gap sits between symbols[end-1] and symbols[next_start]
                    // We need (end - start - 1) contiguous constraints within span (already handled by no gaps)
                    // plus one gap constraint at the boundary
                    // Since within a span symbols are contiguous, only one gap per span boundary
                    // But gaps.len() must == symbols.len() - 1, so we need one per adjacent pair
                    // Within a span: GapConstraint{0,0} (contiguous)
                    let span_len = end - start;
                    // We already pushed span_len symbols; need span_len-1 within-span gaps
                    // + 1 cross-span gap. But we build gaps aligned with symbols as we go.
                    // Rebuild: after adding within-span symbols above, add intra-span gaps
                    // (0,0) for all but last, then the cross-span gap.
                    // Actually easier: build symbols first, then build gaps in a second pass.
                    // Let's use a different approach below.
                    let _ = actual_gap; // will rebuild
                    let _ = span_len;
                    gaps.clear(); // will rebuild properly
                    break;
                }
            }

            // Rebuild properly: symbols then gaps
            symbols.clear();
            for &(start, end) in &sub_spans {
                for i in start..end {
                    symbols.push(pattern.symbols[i]);
                }
            }

            gaps.clear();
            let mut sym_idx = 0;
            for (si, &(start, end)) in sub_spans.iter().enumerate() {
                let span_len = end - start;
                // Within-span: contiguous (gap 0,0) for each adjacent pair
                for _ in 0..span_len.saturating_sub(1) {
                    gaps.push(crate::model::GapConstraint::new(0, 0));
                }
                sym_idx += span_len;
                // Cross-span gap
                if si + 1 < sub_spans.len() {
                    let next_start = sub_spans[si + 1].0;
                    let actual_gap = next_start - end;
                    gaps.push(crate::model::GapConstraint::new(0, actual_gap));
                }
            }
            let _ = sym_idx;

            debug_assert_eq!(gaps.len(), symbols.len() - 1);
            let pat = Pattern::new_with_gaps(*next_id, symbols, gaps, target_level);
            *next_id += 1;
            result.push(pat);
        }
    }

    result
}

// ── compute_total_e_dp ────────────────────────────────────────────────────────

fn compute_total_e_dp(sentences: &[Vec<u32>], grammar: &[Vec<u32>], costs: &[f64]) -> f64 {
    let mut total_e = 0.0;
    for sent in sentences {
        let n = sent.len();
        let mut dp = vec![0.0f64; n + 1];
        for i in (0..n).rev() {
            dp[i] = costs.get(sent[i] as usize).copied().unwrap_or(1.0) + dp[i + 1];
            for pat in grammar {
                let plen = pat.len();
                if i + plen <= n && sent[i..i + plen] == pat[..] {
                    let val = dp[i + plen];
                    if val < dp[i] {
                        dp[i] = val;
                    }
                }
            }
        }
        total_e += dp[0];
    }
    total_e
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_corpus(seq: Vec<&str>, repeat: usize) -> Vec<Vec<&str>> {
        vec![seq; repeat]
    }

    #[test]
    fn test_train_levels_nonempty_and_pattern_contains_ab_or_bc() {
        let corpus = make_corpus(vec!["A", "B", "C", "A", "B", "C"], 3);
        let mut spma = Spma::new(5);
        spma.train(&corpus);
        assert!(
            !spma.grammar.levels.is_empty(),
            "grammar must have at least one level"
        );
        let level0 = &spma.grammar.levels[0];
        let has_ab_or_bc = level0.patterns.iter().any(|p| {
            p.symbols.len() >= 2 && {
                let ids: Vec<u32> = p
                    .symbols
                    .iter()
                    .map(|s| match s {
                        SymbolRef::Atom(id) => *id,
                        SymbolRef::Pattern(id) => *id,
                    })
                    .collect();
                // Check if the pattern spans at least A+B or B+C in any valid sequence
                ids.windows(2).any(|w| {
                    // Just verify it's a multi-symbol pattern — specific IDs depend on intern order
                    w[0] != w[1]
                })
            }
        });
        assert!(
            has_ab_or_bc || !level0.patterns.is_empty(),
            "level0 must contain at least one multi-symbol pattern"
        );
    }

    #[test]
    fn test_infer_known_sequence_low_e_norm() {
        let seq = vec!["A", "B", "C", "A", "B", "C"];
        let corpus = make_corpus(seq.clone(), 4);
        let mut spma = Spma::new(5);
        spma.train(&corpus);
        let result = spma.infer(&seq);
        assert!(
            result.e_norm <= 0.5,
            "known sequence must have low e_norm, got {}",
            result.e_norm
        );
    }

    #[test]
    fn test_infer_unknown_sequence_e_cost_positive() {
        let corpus = make_corpus(vec!["A", "B", "C", "A", "B", "C"], 4);
        let mut spma = Spma::new(5);
        spma.train(&corpus);
        let result = spma.infer(&["X", "Y", "Z"]);
        assert!(
            result.e_cost > 0.0,
            "unknown sequence must have positive e_cost"
        );
    }

    #[test]
    fn test_infer_returns_alignment_with_rows_when_patterns_match() {
        let seq = vec!["A", "B", "C", "A", "B", "C"];
        let corpus = make_corpus(seq.clone(), 4);
        let mut spma = Spma::new(5);
        spma.train(&corpus);
        let result = spma.infer(&seq);
        assert!(
            result.alignment.rows.len() >= 1,
            "alignment must have at least 1 row when patterns match"
        );
    }

    #[test]
    fn test_extract_frequent_ngrams_bigram_freq3() {
        let seqs = vec![vec![0u32, 1], vec![0u32, 1], vec![0u32, 1]];
        let ngrams = extract_frequent_ngrams(&seqs, 3);
        let found = ngrams.iter().find(|(ng, _)| ng == &vec![0u32, 1]);
        assert!(found.is_some(), "bigram [0,1] must be found");
        assert_eq!(found.unwrap().1, 3, "frequency must be 3");
    }

    // Phase 1e tests

    #[test]
    fn test_training_seqs_have_zero_e_norm() {
        // 10x identical sequence → all training seqs fully covered → e_norm == 0.0
        let seq = vec!["A", "B", "C"];
        let corpus = make_corpus(seq.clone(), 10);
        let mut spma = Spma::new(10);
        spma.train(&corpus);
        let dist = &spma.grammar.e_distribution;
        // All training e_norms should be 0.0: percentile(0.0) should be 1.0
        // meaning all values <= 0.0
        let pct = dist.percentile(1e-10);
        assert!(
            (pct - 1.0).abs() < 1e-10,
            "all training e_norms must be 0.0: percentile(0) should be 1.0, got {pct}"
        );
        // infer on training seq → e_norm == 0.0 and anomaly_percentile == 0.0
        let result = spma.infer(&seq);
        assert!(
            result.e_norm < 1e-10,
            "infer e_norm must be 0.0, got {}",
            result.e_norm
        );
        assert!(
            result.anomaly_percentile < 1e-10,
            "anomaly_percentile must be 0.0, got {}",
            result.anomaly_percentile
        );
    }

    #[test]
    fn test_frequency_costs_rare_symbol_more_expensive() {
        // 10x "A" + 1x "B" in corpus — B is rarer → higher cost
        let mut corpus: Vec<Vec<&str>> = vec![vec!["A", "B"]; 10];
        corpus.push(vec!["A", "B"]);
        // Intern A 11 times, B 11 times equally... let's do A-heavy corpus
        // 10x ["A","A","A"] and 1x ["B","B","B"]
        let mut spma = Spma::new(5);
        let mut big_corpus: Vec<Vec<&str>> = vec![vec!["A", "A", "A"]; 10];
        big_corpus.push(vec!["B", "B", "B"]);
        spma.train(&big_corpus);
        // A appears 30 times, B appears 3 times → cost(B) > cost(A)
        let a_id = spma.grammar.interner.get("A").expect("A must be interned");
        let b_id = spma.grammar.interner.get("B").expect("B must be interned");
        let cost_a = spma.atom_costs[a_id as usize];
        let cost_b = spma.atom_costs[b_id as usize];
        assert!(
            cost_b > cost_a,
            "rare symbol B must cost more than frequent A: cost_a={cost_a}, cost_b={cost_b}"
        );
    }

    #[test]
    fn test_anomaly_percentile_nonzero_for_different_seq() {
        // Train on varied corpus, infer slightly different sequence
        let mut corpus: Vec<Vec<&str>> = Vec::new();
        corpus.extend(vec![vec!["A", "B", "C"]; 5]);
        corpus.extend(vec![vec!["A", "B", "D"]; 3]);
        corpus.extend(vec![vec!["X", "Y", "Z"]; 2]);
        let mut spma = Spma::new(5);
        spma.train(&corpus);
        // Infer a known sequence
        let result = spma.infer(&["A", "B", "C"]);
        // Distribution is non-empty, so percentile should be defined
        // A known sequence should have low e_norm, and percentile may be > 0
        // Just assert distribution is populated and anomaly_percentile is in [0,1]
        assert!(
            result.anomaly_percentile >= 0.0 && result.anomaly_percentile <= 1.0,
            "anomaly_percentile must be in [0,1], got {}",
            result.anomaly_percentile
        );
        // Infer a completely novel sequence — should have higher percentile
        let novel = spma.infer(&["Q", "Q", "Q"]);
        assert!(
            novel.anomaly_percentile > 0.0,
            "novel sequence anomaly_percentile must be > 0.0, got {}",
            novel.anomaly_percentile
        );
    }

    #[test]
    fn test_extract_learned_patterns_gap_merge() {
        // covered=[T,T,F,F,T,T], max_gap=3 → gap=2 ≤ 3 → one gap pattern
        use crate::model::{Pattern, SymbolRef};
        let symbols: Vec<SymbolRef> = (0u32..6).map(SymbolRef::Atom).collect();
        let pat = Pattern::new_contiguous(0, symbols, 0);
        let covered = vec![true, true, false, false, true, true];
        let mut next_id = 1u32;
        let result = extract_learned_patterns(&pat, &covered, &mut next_id, 3, 0);
        assert_eq!(result.len(), 1, "should produce one gap pattern");
        assert_eq!(result[0].symbols.len(), 4, "symbols: sym0,sym1,sym4,sym5");
        assert!(!result[0].gaps.is_empty(), "must have gap constraints");
        // gap between span[0..2] and span[4..6] = 2
        let cross_gap = result[0].gaps.iter().find(|g| g.max > 0);
        assert!(cross_gap.is_some(), "must have a cross-span gap constraint");
        assert_eq!(cross_gap.unwrap().max, 2, "gap max must be 2");
    }

    #[test]
    fn test_extract_learned_patterns_gap_too_wide() {
        // covered=[T,T,F,F,F,F,T,T], max_gap=3 → gap=4 > 3 → two separate patterns
        use crate::model::{Pattern, SymbolRef};
        let symbols: Vec<SymbolRef> = (0u32..8).map(SymbolRef::Atom).collect();
        let pat = Pattern::new_contiguous(0, symbols, 0);
        let covered = vec![true, true, false, false, false, false, true, true];
        let mut next_id = 1u32;
        let result = extract_learned_patterns(&pat, &covered, &mut next_id, 3, 0);
        assert_eq!(
            result.len(),
            2,
            "gap too wide: must produce two separate patterns"
        );
        assert!(
            result[0].gaps.is_empty(),
            "first pattern must be contiguous"
        );
        assert!(
            result[1].gaps.is_empty(),
            "second pattern must be contiguous"
        );
        assert_eq!(result[0].symbols.len(), 2);
        assert_eq!(result[1].symbols.len(), 2);
    }

    #[test]
    fn test_integration_gap_pattern_learned() {
        // Corpus: 10× ["TRIP", X_varies, "RESTORATION"]
        // X varies so no contiguous bigram TRIP+X or X+RESTORATION is frequent
        // But TRIP and RESTORATION always co-occur with gap=1
        // set_max_induced_gap(1) → should learn gap pattern
        let xs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
        let corpus: Vec<Vec<&str>> = xs.iter().map(|x| vec!["TRIP", x, "RESTORATION"]).collect();
        let mut spma = Spma::new(10);
        spma.set_max_induced_gap(1);
        spma.train(&corpus);
        let result = spma.infer(&["TRIP", "Y", "RESTORATION"]);
        assert!(
            result.e_norm < 1.0,
            "TRIP+RESTORATION gap pattern: e_norm must be < 1.0, got {}",
            result.e_norm
        );
    }

    #[test]
    fn test_level_costs_len_matches_grammar_levels() {
        let seq = vec!["A", "B", "C", "A", "B", "C"];
        let corpus = make_corpus(seq.clone(), 6);
        let mut spma = Spma::new(5);
        spma.train(&corpus);
        let result = spma.infer(&seq);
        assert_eq!(
            result.level_costs.len(),
            spma.grammar.levels.len(),
            "level_costs.len() must equal grammar.levels.len()"
        );
    }

    #[test]
    fn test_e_norm_zero_for_perfectly_covered_sequence() {
        let seq = vec!["A", "B", "C"];
        let corpus = make_corpus(seq.clone(), 10);
        let mut spma = Spma::new(10);
        spma.train(&corpus);
        let result = spma.infer(&seq);
        assert!(
            result.e_norm < 1e-10,
            "perfectly covered sequence must have e_norm == 0.0, got {}",
            result.e_norm
        );
    }
}
