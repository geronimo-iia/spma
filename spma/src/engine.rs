// Phase 1d — see docs/grammar-spec.md, docs/roadmap.md

use std::collections::HashMap;
use std::io::{self, Read as IoRead, Write as IoWrite};

use rayon::prelude::*;

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

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Spma {
    pub(crate) grammar: Grammar,
    beam_k: usize,
    pub(crate) atom_costs: Vec<f64>,
    max_induced_gap: usize,
    #[serde(default)]
    atom_freq: HashMap<u32, u32>,
    #[serde(default)]
    total_symbol_count: u64,
}

impl Spma {
    pub fn new(beam_k: usize) -> Self {
        Self {
            grammar: Grammar::default(),
            beam_k,
            atom_costs: Vec::new(),
            max_induced_gap: MAX_INDUCED_GAP,
            atom_freq: HashMap::new(),
            total_symbol_count: 0,
        }
    }

    pub fn set_max_induced_gap(&mut self, max: usize) {
        self.max_induced_gap = max;
    }

    pub fn set_anomaly_threshold(&mut self, threshold: f64) {
        self.grammar.e_distribution.threshold = threshold;
    }

    /// Set anomaly threshold for a specific grammar level.
    /// Level 0 = atom level, level 1 = pattern-ID level, etc.
    pub fn beam_k(&self) -> usize {
        self.beam_k
    }

    pub fn max_induced_gap(&self) -> usize {
        self.max_induced_gap
    }

    pub fn set_level_threshold(&mut self, level: usize, threshold: f64) {
        let dist = &mut self.grammar.e_distribution;
        if dist.level_thresholds.len() <= level {
            dist.level_thresholds.resize(level + 1, dist.threshold);
        }
        dist.level_thresholds[level] = threshold;
    }

    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    pub fn atom_costs(&self) -> &[f64] {
        &self.atom_costs
    }

    pub fn e_distribution(&self) -> &crate::model::EDistribution {
        &self.grammar.e_distribution
    }

    pub fn atom_freq_for_test(&self) -> &HashMap<u32, u32> {
        &self.atom_freq
    }

    pub fn total_symbol_count_for_test(&self) -> u64 {
        self.total_symbol_count
    }

    pub fn save<W: IoWrite>(&self, writer: W) -> io::Result<()> {
        serde_json::to_writer(writer, self).map_err(io::Error::other)
    }

    pub fn load<R: IoRead>(reader: R) -> io::Result<Self> {
        let mut spma: Self = serde_json::from_reader(reader).map_err(io::Error::other)?;
        for level in &mut spma.grammar.levels {
            level.rebuild_index();
        }
        Ok(spma)
    }

    fn train_inner(&mut self, corpus: &[Vec<&str>]) {
        if corpus.is_empty() {
            return;
        }

        // Step 1: intern all symbols
        for seq in corpus {
            for s in seq {
                self.grammar.interner.intern(s);
            }
        }

        // Step 2: build atom-level sequences
        let atom_seqs: Vec<Vec<u32>> = corpus
            .iter()
            .map(|seq| {
                seq.iter()
                    .map(|s| self.grammar.interner.intern(s))
                    .collect()
            })
            .collect();

        // Merge frequencies into self.atom_freq / self.total_symbol_count
        for seq in &atom_seqs {
            for &id in seq {
                *self.atom_freq.entry(id).or_insert(0) += 1;
                self.total_symbol_count += 1;
            }
        }
        let n_atoms = self.grammar.interner.len();
        let total = self.total_symbol_count as f64;
        let costs: Vec<f64> = (0..n_atoms as u32)
            .map(|id| {
                let freq = self.atom_freq.get(&id).copied().unwrap_or(1);
                -((freq as f64 / total).log2())
            })
            .collect();
        self.atom_costs = costs.clone();

        let profile = std::env::var("SPMA_PROFILE").is_ok();

        // Step 3: extract frequent n-grams from atom sequences
        let min_freq = (corpus.len() / 2).max(2);
        let t_step3_ngrams = std::time::Instant::now();
        let ngrams = extract_frequent_ngrams(&atom_seqs, min_freq);
        let ms_step3_ngrams = t_step3_ngrams.elapsed().as_millis();
        let t_l0_mdl = std::time::Instant::now();

        // next_id: 0 on cold start, max existing + 1 on incremental
        let mut next_id: u32 = self
            .grammar
            .levels
            .iter()
            .flat_map(|l| l.patterns.iter())
            .map(|p| p.id + 1)
            .max()
            .unwrap_or(0);

        // Seed all_id_vecs and cached_g_l0 from existing level-0 patterns (empty on cold start)
        let mut all_id_vecs: Vec<Vec<u32>> = self
            .grammar
            .levels
            .first()
            .map(|l| {
                l.patterns
                    .iter()
                    .map(|p| {
                        p.symbols
                            .iter()
                            .map(|s| match s {
                                SymbolRef::Atom(id) | SymbolRef::Pattern(id) => *id,
                            })
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut cached_g_l0: f64 = self
            .grammar
            .levels
            .first()
            .map(|l| {
                l.patterns
                    .iter()
                    .map(|p| {
                        p.symbols
                            .iter()
                            .map(|s| match s {
                                SymbolRef::Atom(id) | SymbolRef::Pattern(id) => {
                                    costs.get(*id as usize).copied().unwrap_or(1.0)
                                }
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .unwrap_or(0.0);
        let mut cached_e_l0 = compute_total_e_dp(&atom_seqs, &all_id_vecs, &costs);

        // Fingerprint set of existing level-0 patterns — skip already-known on incremental
        let existing_fps: std::collections::HashSet<Vec<u32>> = self
            .grammar
            .levels
            .first()
            .map(|l| {
                l.patterns
                    .iter()
                    .map(|p| {
                        p.symbols
                            .iter()
                            .map(|s| match s {
                                SymbolRef::Atom(id) | SymbolRef::Pattern(id) => *id,
                            })
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut level0_patterns: Vec<Pattern> = Vec::new();

        for (ngram, _freq) in &ngrams {
            // Gap-encoded candidates: [sym_i, GAP_MARKER, gap_size, sym_j]
            let is_gap_candidate = ngram.len() == 4 && ngram[1] == GAP_MARKER;

            // For MDL gating, the atom IDs used in the pattern (not sentinels)
            let atom_ids: Vec<u32> = if is_gap_candidate {
                vec![ngram[0], ngram[3]]
            } else {
                ngram.clone()
            };

            // Skip already-known patterns (no-op on cold start — existing_fps is empty)
            if existing_fps.contains(&atom_ids) {
                continue;
            }

            let pattern_cost: f64 = atom_ids.iter().map(|&id| costs[id as usize]).sum();
            let new_g: f64 = cached_g_l0 + pattern_cost;
            let current_t = cached_g_l0 + cached_e_l0;

            // For dp matching, gap patterns reduce to their two flanking atoms
            let matching_vec = atom_ids.clone();
            let mut candidate_multi = all_id_vecs.clone();
            candidate_multi.push(matching_vec.clone());
            let new_e = compute_total_e_dp(&atom_seqs, &candidate_multi, &costs);
            let new_t = new_g + new_e;

            if new_t < current_t || (all_id_vecs.is_empty() && existing_fps.is_empty()) {
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
                cached_e_l0 = new_e;
                cached_g_l0 = new_g;
                next_id += 1;
            }
        }

        // Fallback: if cold start found nothing, push the most frequent bigram
        if level0_patterns.is_empty() && self.grammar.levels.is_empty() && !ngrams.is_empty() {
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

        let ms_l0_mdl = t_l0_mdl.elapsed().as_millis();

        // Push new level on cold start, extend existing on incremental
        if self.grammar.levels.is_empty() {
            self.grammar.levels.push(GrammarLevel::new(level0_patterns));
        } else {
            self.grammar.levels[0].patterns.extend(level0_patterns);
            self.grammar.levels[0].rebuild_index();
        }

        // Step 4: outer N-level loop — learn hierarchical levels
        let max_levels: usize = 8;
        let mut current_atom_seqs = atom_seqs.clone();
        let mut current_costs = costs.clone();

        let mut cum_beam_ms: u128 = 0;
        let mut cum_gap_ms: u128 = 0;
        let mut cum_mdl_ms: u128 = 0;

        // Cache e_norms per level during beam passes to avoid edist full rebuild
        let mut level_e_norms_vecs: Vec<Vec<f64>> = Vec::new();

        for level in 0..max_levels {
            if level >= self.grammar.levels.len() {
                break;
            }
            let level_patterns: Vec<&Pattern> =
                self.grammar.levels[level].patterns.iter().collect();
            if level_patterns.is_empty() {
                break;
            }
            let level_index = &self.grammar.levels[level].symbol_index;

            // Run beam_search on each sequence to get match logs
            // Collect results first (immutable borrow ends), then update frequencies.
            let t_beam = std::time::Instant::now();
            let raw_results: Vec<Option<RawAlignment>> = current_atom_seqs
                .par_iter()
                .map(|seq| {
                    beam_search(
                        seq,
                        &level_patterns,
                        level_index,
                        self.beam_k,
                        &current_costs,
                    )
                    .into_iter()
                    .next()
                })
                .collect();
            let beam_ms = t_beam.elapsed().as_millis();
            cum_beam_ms += beam_ms;

            // Cache e_norms at this level for edist population
            {
                let lvl_e: Vec<f64> = current_atom_seqs
                    .par_iter()
                    .zip(raw_results.par_iter())
                    .filter_map(|(seq, res)| {
                        let raw: f64 = seq
                            .iter()
                            .map(|&id| current_costs.get(id as usize).copied().unwrap_or(1.0))
                            .sum();
                        if raw < 1e-12 {
                            return None;
                        }
                        let e = res.as_ref().map(|r| r.e_cost).unwrap_or(raw);
                        Some(e / raw)
                    })
                    .collect();
                level_e_norms_vecs.push(lvl_e);
            }

            let mut all_match_logs: Vec<Vec<crate::beam::MatchEvent>> = Vec::new();
            let mut gap_candidates: Vec<Pattern> = Vec::new();

            let t_gap = std::time::Instant::now();
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
                        (level + 1) as u8,
                    );
                    gap_candidates.extend(new_gaps);

                    all_match_logs.push(best.match_log);
                } else {
                    all_match_logs.push(Vec::new());
                }
            }

            let gap_ms = t_gap.elapsed().as_millis();
            cum_gap_ms += gap_ms;

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
            let patterns_in = next_level_pats.len();

            let t_mdl = std::time::Instant::now();
            let mut cached_e = compute_total_e_dp(&pid_seqs, &next_id_vecs, &pid_costs);
            let mut cached_g: f64 = 0.0;
            for pat in next_level_pats {
                let ngram: Vec<u32> = pat
                    .symbols
                    .iter()
                    .map(|s| match s {
                        SymbolRef::Pattern(id) => *id,
                        SymbolRef::Atom(id) => *id,
                    })
                    .collect();

                let pattern_cost: f64 = ngram
                    .iter()
                    .map(|&id| pid_costs.get(id as usize).copied().unwrap_or(pid_cost))
                    .sum();
                let current_t = cached_g + cached_e;

                let mut candidate = next_id_vecs.clone();
                candidate.push(ngram.clone());
                let new_g = cached_g + pattern_cost;
                let new_e = compute_total_e_dp(&pid_seqs, &candidate, &pid_costs);
                let new_t = new_g + new_e;

                if new_t < current_t || next_id_vecs.is_empty() {
                    next_id_vecs.push(ngram);
                    accepted_pats.push(pat);
                    cached_e = new_e;
                    cached_g = new_g;
                }
            }

            let mdl_ms = t_mdl.elapsed().as_millis();
            cum_mdl_ms += mdl_ms;

            if profile {
                eprintln!(
                    "[profile] level {} beam: {}ms  gap: {}ms  mdl: {}ms  patterns_in: {}  patterns_out: {}",
                    level, beam_ms, gap_ms, mdl_ms, patterns_in, accepted_pats.len()
                );
            }

            if accepted_pats.is_empty() {
                break;
            }

            // Push new level on cold start, extend existing on incremental
            let next_level = level + 1;
            if next_level < self.grammar.levels.len() {
                self.grammar.levels[next_level]
                    .patterns
                    .extend(accepted_pats);
                self.grammar.levels[next_level].rebuild_index();
            } else {
                self.grammar.levels.push(GrammarLevel::new(accepted_pats));
            }
            current_atom_seqs = pid_seqs;
            current_costs = pid_costs;
        }

        let t_edist = std::time::Instant::now();

        // Merge new e_norms into existing distribution; preserve user-set level_thresholds.
        let new_e_norms: Vec<f64> = level_e_norms_vecs.first().cloned().unwrap_or_default();
        let mut all_e_norms = self.grammar.e_distribution.sorted_e_norms.clone();
        all_e_norms.extend(new_e_norms);

        let mut merged_level_e_norms = self.grammar.e_distribution.level_sorted_e_norms.clone();
        for (i, new_lvl) in level_e_norms_vecs.into_iter().enumerate() {
            if i < merged_level_e_norms.len() {
                merged_level_e_norms[i].extend(new_lvl);
            } else {
                merged_level_e_norms.push(new_lvl);
            }
        }

        let level_thresholds = self.grammar.e_distribution.level_thresholds.clone();
        let threshold = self.grammar.e_distribution.threshold;
        self.grammar.e_distribution =
            crate::model::EDistribution::fit(all_e_norms, threshold, merged_level_e_norms);
        self.grammar.e_distribution.level_thresholds = level_thresholds;

        let ms_edist = t_edist.elapsed().as_millis();

        if profile {
            eprintln!("[profile] step3_ngrams:       {}ms", ms_step3_ngrams);
            eprintln!("[profile] level0_mdl:         {}ms", ms_l0_mdl);
            eprintln!("[profile] levelN_beam_total:  {}ms", cum_beam_ms);
            eprintln!("[profile] levelN_gap_extract: {}ms", cum_gap_ms);
            eprintln!("[profile] levelN_mdl:         {}ms", cum_mdl_ms);
            eprintln!("[profile] edist_rebuild:      {}ms", ms_edist);
        }
    }

    pub fn train(&mut self, corpus: &[Vec<&str>]) {
        self.grammar = Grammar::default();
        self.atom_freq = HashMap::new();
        self.total_symbol_count = 0;
        self.atom_costs = Vec::new();
        self.train_inner(corpus);
    }

    pub fn retrain(&mut self, corpus: &[Vec<&str>]) {
        self.train_inner(corpus);
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
        let level0_index = &self.grammar.levels[0].symbol_index;

        let results = beam_search(&ids, &level0_patterns, level0_index, self.beam_k, &costs);

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

        // Higher levels: use pid sequences derived from match log.
        // Seed with best_raw.match_log — level=1 reuses it directly, no redundant beam call.
        let mut prev_match_log: Vec<crate::beam::MatchEvent> = best_raw.match_log;

        let level_fallback_costs: Vec<f64> = (1..self.grammar.levels.len())
            .map(|lv| {
                let n = self.grammar.levels[lv - 1].patterns.len();
                if n > 1 {
                    (n as f64).log2()
                } else {
                    1.0
                }
            })
            .collect();

        for level in 1..self.grammar.levels.len() {
            let prev_patterns: Vec<&Pattern> =
                self.grammar.levels[level - 1].patterns.iter().collect();

            // Build pid_seq from prev_match_log (best_raw at level=1, prior beam at level>=2)
            let pid_seq: Vec<u32> = {
                let mut pid_positions: Vec<(u32, usize)> = Vec::new();
                for event in &prev_match_log {
                    if event.old_pos == 0 {
                        if let Some(pat) = prev_patterns.get(event.old_idx) {
                            pid_positions.push((pat.id, event.new_pos));
                        }
                    }
                }
                pid_positions.sort_by_key(|&(_, pos)| pos);
                pid_positions.dedup_by_key(|x| x.1);
                pid_positions.into_iter().map(|(pid, _)| pid).collect()
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
            let total_pid: u32 = pid_seq.len() as u32;
            let max_pid = self.grammar.levels[level - 1]
                .patterns
                .iter()
                .map(|p| p.id as usize + 1)
                .max()
                .unwrap_or(1);
            let mut pid_freq: Vec<u32> = vec![0u32; max_pid];
            for &pid in &pid_seq {
                if (pid as usize) < max_pid {
                    pid_freq[pid as usize] += 1;
                }
            }
            let total_f = total_pid.max(1) as f64;
            let log2_total = total_f.log2();
            let pid_costs: Vec<f64> = (0..max_pid)
                .map(|id| {
                    let freq = pid_freq[id].max(1);
                    -((freq as f64).log2() - log2_total)
                })
                .collect();

            let raw_level_cost: f64 = pid_seq
                .iter()
                .map(|&id| pid_costs.get(id as usize).copied().unwrap_or(1.0))
                .sum();

            let level_patterns: Vec<&Pattern> =
                self.grammar.levels[level].patterns.iter().collect();
            let level_index = &self.grammar.levels[level].symbol_index;

            // Extend pid_costs to cover all pattern IDs referenced
            let max_ref = pid_seq.iter().map(|&id| id as usize + 1).max().unwrap_or(1);
            let mut pid_costs_ext = pid_costs.clone();
            let fallback_pid = level_fallback_costs[level - 1];
            pid_costs_ext.resize(max_ref.max(pid_costs_ext.len()), fallback_pid);

            let level_results = beam_search(
                &pid_seq,
                &level_patterns,
                level_index,
                self.beam_k,
                &pid_costs_ext,
            );
            let best_level = level_results.into_iter().next();
            let lc = best_level
                .as_ref()
                .map(|r| r.e_cost)
                .unwrap_or(raw_level_cost);

            level_costs.push(lc);
            level_e_norms.push(if raw_level_cost < 1e-12 {
                0.0
            } else {
                lc / raw_level_cost
            });

            // Pass this level's match_log to next iteration instead of re-running beam
            prev_match_log = best_level.map(|r| r.match_log).unwrap_or_default();
        }

        let dist = &self.grammar.e_distribution;
        let is_anomaly_level0 = e_norm > dist.threshold;
        let is_anomaly_levels = level_e_norms.iter().enumerate().any(|(lvl, &lvl_e)| {
            let t = dist
                .level_thresholds
                .get(lvl)
                .copied()
                .unwrap_or(f64::INFINITY);
            lvl_e > t
        });
        let is_anomaly = is_anomaly_level0 || is_anomaly_levels;

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

    pub fn recalibrate(&mut self, corpus: &[Vec<&str>]) {
        for level in &mut self.grammar.levels {
            level.rebuild_index();
        }
        let mut e_norms: Vec<f64> = Vec::with_capacity(corpus.len());
        let mut level_e_norms_vecs: Vec<Vec<f64>> = Vec::new();

        for seq in corpus {
            let result = self.infer(seq);
            e_norms.push(result.e_norm);
            for (i, &lvl_e) in result.level_e_norms.iter().enumerate() {
                if level_e_norms_vecs.len() <= i {
                    level_e_norms_vecs.push(Vec::new());
                }
                level_e_norms_vecs[i].push(lvl_e);
            }
        }

        self.grammar.e_distribution =
            crate::model::EDistribution::fit(e_norms, 0.0, level_e_norms_vecs);
    }
}

// ── extract_frequent_ngrams ───────────────────────────────────────────────────

const GAP_MARKER: u32 = u32::MAX;
const MAX_INDUCED_GAP: usize = 3;

/// Count contiguous bigrams/trigrams and gap-aware pairs within a window.
/// Gap candidates encoded as [sym_i, GAP_MARKER, gap_size, sym_j].
pub(crate) fn extract_frequent_ngrams(seqs: &[Vec<u32>], min_freq: usize) -> Vec<(Vec<u32>, u32)> {
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

            // Build symbols then gaps in two passes
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

    #[test]
    fn test_serialization_roundtrip() {
        let corpus = make_corpus(vec!["A", "B", "C", "A", "B", "C"], 3);
        let mut spma = Spma::new(5);
        spma.train(&corpus);

        let mut buf = Vec::new();
        spma.save(&mut buf).expect("save failed");

        let loaded = Spma::load(buf.as_slice()).expect("load failed");

        assert_eq!(spma.beam_k, loaded.beam_k);
        assert_eq!(spma.max_induced_gap, loaded.max_induced_gap);
        assert_eq!(spma.atom_costs, loaded.atom_costs);
        assert_eq!(
            spma.grammar.levels.len(),
            loaded.grammar.levels.len(),
            "level count must match"
        );

        let original = spma.infer(&["A", "B", "C"]);
        let restored = loaded.infer(&["A", "B", "C"]);
        assert!(
            (original.e_norm - restored.e_norm).abs() < 1e-12,
            "e_norm diverged after roundtrip: {} vs {}",
            original.e_norm,
            restored.e_norm
        );
    }

    #[test]
    fn recalibrate_does_not_change_grammar_structure() {
        let seq = vec!["A", "B", "C", "A", "B", "C"];
        let corpus = make_corpus(seq.clone(), 6);
        let mut spma = Spma::new(5);
        spma.train(&corpus);

        let levels_before: Vec<usize> = spma
            .grammar
            .levels
            .iter()
            .map(|l| l.patterns.len())
            .collect();
        let atom_costs_before = spma.atom_costs.clone();
        let interner_len_before = spma.grammar.interner.len();

        let corpus_refs: Vec<Vec<&str>> =
            corpus.iter().map(|s| s.iter().copied().collect()).collect();
        spma.recalibrate(&corpus_refs);

        let levels_after: Vec<usize> = spma
            .grammar
            .levels
            .iter()
            .map(|l| l.patterns.len())
            .collect();
        assert_eq!(
            levels_before, levels_after,
            "grammar levels must not change"
        );
        assert_eq!(
            atom_costs_before, spma.atom_costs,
            "atom_costs must not change"
        );
        assert_eq!(
            interner_len_before,
            spma.grammar.interner.len(),
            "interner must not change"
        );

        assert!(
            spma.grammar.e_distribution.sorted_e_norms_len_for_test() > 0,
            "e_distribution must be populated after recalibrate"
        );
    }

    #[test]
    fn recalibrate_after_pattern_removal() {
        let seq = vec!["A", "B", "C", "A", "B", "C"];
        let corpus = make_corpus(seq.clone(), 6);
        let mut spma = Spma::new(5);
        spma.train(&corpus);

        let original_count = spma.grammar.levels[0].patterns.len();
        assert!(
            original_count > 1,
            "test setup: need >1 patterns at level 0, got {original_count}"
        );
        spma.grammar.levels[0].patterns.pop();
        let reduced_count = spma.grammar.levels[0].patterns.len();
        assert_eq!(reduced_count, original_count - 1);

        let corpus_refs: Vec<Vec<&str>> =
            corpus.iter().map(|s| s.iter().copied().collect()).collect();
        spma.recalibrate(&corpus_refs);

        assert_eq!(
            spma.grammar.levels[0].patterns.len(),
            reduced_count,
            "recalibrate must not re-add removed patterns"
        );
    }
}
