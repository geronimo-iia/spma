//! SPMA learning engine — pattern recognition, grammar extraction, and compression
//! through multiple alignment construction (J G Wolff's framework).

use crate::*;
use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarLevel {
    pub old_patterns: Vec<Pattern>,
    pub corpus_costs: Vec<f64>,
}

fn collect_frequencies(freqs: &mut HashMap<u32, u32>, patterns: &[Pattern]) {
    for pattern in patterns {
        for symbol in &pattern.symbols {
            *freqs.entry(symbol.raw_id()).or_insert(0) += pattern.frequency;
        }
    }
}

fn apply_symbol_costs(freqs: &HashMap<u32, u32>, patterns: &mut [Pattern]) {
    let total_freq: u32 = freqs.values().sum();
    for pattern in patterns {
        for symbol in &mut pattern.symbols {
            if let Some(&freq) = freqs.get(&symbol.raw_id()) {
                if freq > 0 && total_freq > 0 {
                    let probability = freq as f64 / total_freq as f64;
                    symbol.bit_cost = -probability.log2();
                } else {
                    symbol.bit_cost = 4.0;
                }
            }
        }
    }
}

pub fn write_alignment_table(
    w: &mut impl FmtWrite,
    new_pattern: &Pattern,
    alignment: &BeamAlignment,
    old_patterns: &[Pattern],
    interner: &Interner,
) {
    let new_syms: Vec<String> = new_pattern.get_symbol_names(interner);
    let n = new_syms.len();
    if n == 0 {
        return;
    }

    let used_olds: Vec<(&Pattern, Vec<String>)> = alignment
        .old_pattern_indices
        .iter()
        .filter_map(|&i| old_patterns.get(i))
        .map(|p| (p, p.get_symbol_names(interner)))
        .collect();

    let mut assignment: Vec<Option<usize>> = vec![None; n];
    for (row_idx, (_, old_names)) in used_olds.iter().enumerate() {
        for (p, covered) in alignment.covered_new.iter().enumerate() {
            if *covered && assignment[p].is_none() && old_names.contains(&new_syms[p]) {
                assignment[p] = Some(row_idx);
            }
        }
    }

    let col_widths: Vec<usize> = (0..n)
        .map(|p| {
            let base = new_syms[p].len();
            let old_max = used_olds
                .iter()
                .map(|(_, names)| {
                    if names.contains(&new_syms[p]) { new_syms[p].len() } else { 1 }
                })
                .max()
                .unwrap_or(1);
            base.max(old_max) + 2
        })
        .collect();

    let label_width = used_olds
        .len()
        .checked_sub(1)
        .map(|last| format!("Old{}:", last + 1).len())
        .unwrap_or(4)
        .max("New:".len());

    let _ = write!(w, "{:<width$}", "New:", width = label_width + 1);
    for (p, sym) in new_syms.iter().enumerate() {
        let _ = write!(w, "{:<width$}", sym, width = col_widths[p]);
    }
    let _ = writeln!(w);

    for (row_idx, (_, old_names)) in used_olds.iter().enumerate() {
        let label = format!("Old{}:", row_idx + 1);
        let _ = write!(w, "{:<width$}", label, width = label_width + 1);
        for (p, _) in new_syms.iter().enumerate() {
            let cell = if alignment.covered_new[p] && assignment[p] == Some(row_idx) {
                old_names
                    .iter()
                    .find(|&s| s == &new_syms[p])
                    .cloned()
                    .unwrap_or_else(|| " ".to_string())
            } else if alignment.covered_new[p] && assignment[p].is_none() {
                "-".to_string()
            } else {
                " ".to_string()
            };
            let _ = write!(w, "{:<width$}", cell, width = col_widths[p]);
        }
        let _ = writeln!(w, " [{}]", old_names.join(" "));
    }

    let matched = alignment.covered_new.iter().filter(|&&c| c).count();
    let _ = writeln!(
        w,
        "\nMatched: {}/{}  G={:.1} bits  E={:.1} bits  T={:.1} bits  CD={:+.1} bits",
        matched, n, alignment.g, alignment.e, alignment.t, alignment.cd
    );
    let _ = writeln!(w);
}

pub fn print_alignment_table(
    new_pattern: &Pattern,
    alignment: &BeamAlignment,
    old_patterns: &[Pattern],
    interner: &Interner,
) {
    let mut out = String::new();
    write_alignment_table(&mut out, new_pattern, alignment, old_patterns, interner);
    print!("{out}");
}

/// Main SPMA learning system that discovers patterns and builds grammars
/// through unsupervised learning from input sequences.
///
pub struct SpmaEngine {
    pub interner: Interner,

    pub old_patterns: Vec<Pattern>,
    pub new_patterns: Vec<Pattern>,

    pub symbol_frequencies: HashMap<u32, u32>,
    pub original_alphabet: HashSet<u32>,

    /// Symbol costs derived from the training corpus (new_patterns frequencies).
    /// Indexed by symbol ID. Used at inference as fallback for symbols not in any
    /// grammar pattern (which would otherwise have cost=0, masking uncovered symbols).
    pub corpus_costs: Vec<f64>,

    pub next_pattern_id: u32,

    pub verbose: bool,

    pub max_cycles: u32,
    pub keep_rows: u32,
    pub grammar_levels: Vec<GrammarLevel>,
    pub max_levels_safety_cap: u8,
}

impl Default for SpmaEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SpmaEngine {
    /// Creates a new SPMA learning system with default parameters.
    ///
    /// # Returns
    ///
    /// A new `SpmaEngine` instance ready for learning.
    pub fn new() -> Self {
        Self {
            interner: Interner::new(),
            old_patterns: Vec::new(),
            new_patterns: Vec::new(),
            symbol_frequencies: HashMap::new(),
            original_alphabet: HashSet::new(),
            corpus_costs: Vec::new(),
            next_pattern_id: 1,
            verbose: false,
            max_cycles: 1000,
            keep_rows: 5,
            grammar_levels: Vec::new(),
            max_levels_safety_cap: 16,
        }
    }

    /// Loads input patterns from a text file.
    ///
    /// Each line in the file represents a pattern with space-separated symbols.
    /// Lines starting with '#' are treated as comments and ignored.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the input file
    ///
    /// # Returns
    ///
    /// A vector of parsed patterns ready for learning.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    ///
    pub fn load_input(&mut self, filename: &str) -> Result<Vec<Pattern>> {
        let content = fs::read_to_string(filename)
            .with_context(|| format!("Failed to read input file: {}", filename))?;

        let mut patterns = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let symbol_names: Vec<&str> = line.split_whitespace().collect();
            let mut symbols = Vec::new();

            for (i, name) in symbol_names.iter().enumerate() {
                // Detect symbol types and determine the canonical name to intern
                let (canonical_name, sym_type, sym_status) = match *name {
                    "<" => ("<", SymbolType::LeftBracket, SymbolStatus::BoundaryMarker),
                    ">" => (">", SymbolType::RightBracket, SymbolStatus::BoundaryMarker),
                    n if n.starts_with('#') => {
                        (n, SymbolType::UniqueIdSymbol, SymbolStatus::Identification)
                    }
                    n if n.starts_with('!') => (
                        &n[1..],
                        SymbolType::DataSymbol,
                        SymbolStatus::Identification,
                    ),
                    n => (n, SymbolType::DataSymbol, SymbolStatus::Contents),
                };

                let id = self.interner.intern(canonical_name);
                let mut symbol = Symbol::new(id);
                symbol.position = i as i32;
                symbol.symbol_type = sym_type;
                symbol.status = sym_status;

                symbols.push(symbol);
                self.original_alphabet.insert(id);
            }

            if !symbols.is_empty() {
                let pattern = Pattern::new(symbols, self.next_pattern_id);
                patterns.push(pattern);
                self.next_pattern_id += 1;
            }
        }

        Ok(patterns)
    }

    /// Assigns symbol types based on their usage patterns across all patterns.
    ///
    /// Symbols marked as identification in any pattern become context symbols.
    /// Data symbols get their costs multiplied by the cost factor.
    pub fn assign_symbol_types(&mut self) {
        let mut context_symbols = HashSet::new();

        for pattern in self.old_patterns.iter().chain(self.new_patterns.iter()) {
            for symbol in &pattern.symbols {
                if symbol.status == SymbolStatus::Identification {
                    context_symbols.insert(symbol.raw_id());
                }
            }
        }

        for pattern in self
            .old_patterns
            .iter_mut()
            .chain(self.new_patterns.iter_mut())
        {
            for symbol in &mut pattern.symbols {
                if context_symbols.contains(&symbol.raw_id()) {
                    symbol.symbol_type = SymbolType::ContextSymbol;
                }
            }
        }
    }

    pub fn calculate_symbol_frequencies(&mut self, patterns: &[Pattern]) {
        self.symbol_frequencies.clear();
        collect_frequencies(&mut self.symbol_frequencies, patterns);
    }

    pub fn assign_symbol_costs(&mut self, patterns: &mut [Pattern]) {
        apply_symbol_costs(&self.symbol_frequencies, patterns);
    }

    pub fn run_recognition_cycle_beam(&mut self, new_pattern: &Pattern) -> Option<BeamAlignment> {
        let max_id = self.interner.len().max(self.next_pattern_id as usize + 1);
        let mut costs = vec![0.0f64; max_id];
        for p in &self.old_patterns {
            for s in &p.symbols {
                if (s.raw_id() as usize) < costs.len() {
                    costs[s.raw_id() as usize] = s.bit_cost;
                }
            }
        }
        for s in &new_pattern.symbols {
            if (s.raw_id() as usize) < costs.len() {
                costs[s.raw_id() as usize] = s.bit_cost;
            }
        }

        let new_ids: Vec<u32> = new_pattern.symbols.iter().map(|s| s.raw_id()).collect();
        let old_id_vecs: Vec<Vec<u32>> = self
            .old_patterns
            .iter()
            .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
            .collect();

        let alignments = beam_search(&new_ids, &old_id_vecs, self.keep_rows as usize, &costs);
        alignments.into_iter().next()
    }

    pub fn learn(&mut self, input_patterns: Vec<Pattern>) -> Result<LearningResults> {
        self.new_patterns = input_patterns;
        self.old_patterns.clear();
        self.grammar_levels.clear();

        // Initial cost assignment
        self.symbol_frequencies.clear();
        collect_frequencies(&mut self.symbol_frequencies, &self.new_patterns);
        apply_symbol_costs(&self.symbol_frequencies, &mut self.new_patterns);
        self.assign_symbol_types();

        let mut total_cycles = 0u32;

        loop {
            total_cycles += 1;

            let old_count_before = self.old_patterns.len();

            // Pass 1: collect candidate patterns from beam alignments (count occurrences)
            let new_patterns_snapshot = self.new_patterns.clone();
            let mut candidates: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
            for new_pattern in &new_patterns_snapshot {
                let best_opt = self.run_recognition_cycle_beam(new_pattern);

                if let Some(best) = best_opt {
                    if best.cd > 0.0 {
                        // Increment frequency of used Old patterns
                        for &oi in &best.old_pattern_indices {
                            if oi < self.old_patterns.len() {
                                self.old_patterns[oi].frequency += 1;
                            }
                        }

                        for learned in extract_learned_patterns(
                            new_pattern,
                            &best.covered_new,
                            &mut self.next_pattern_id,
                        ) {
                            let learned_ids: Vec<u32> =
                                learned.symbols.iter().map(|s| s.raw_id()).collect();
                            *candidates.entry(learned_ids).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Pass 2: apply MDL check — only add candidates that reduce global T
            {
                let max_id = self.interner.len();
                let mut costs = vec![0.0f64; max_id];
                for p in self.old_patterns.iter().chain(self.new_patterns.iter()) {
                    for s in &p.symbols {
                        if (s.raw_id() as usize) < max_id {
                            costs[s.raw_id() as usize] = s.bit_cost;
                        }
                    }
                }
                let new_id_vecs: Vec<Vec<u32>> = self
                    .new_patterns
                    .iter()
                    .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
                    .collect();

                let mut sorted_candidates: Vec<(Vec<u32>, u32)> = candidates.into_iter().collect();
                sorted_candidates.sort_by(|a, b| {
                    let cost_a: f64 = a.0.iter().map(|&id| costs[id as usize]).sum();
                    let cost_b: f64 = b.0.iter().map(|&id| costs[id as usize]).sum();
                    let save_a = (a.1 as f64 - 1.0) * cost_a;
                    let save_b = (b.1 as f64 - 1.0) * cost_b;
                    save_b
                        .partial_cmp(&save_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });

                let mut current_multi: Vec<Vec<u32>> = self
                    .old_patterns
                    .iter()
                    .filter(|p| p.symbols.len() >= 2)
                    .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
                    .collect();
                let mut current_g: f64 = current_multi
                    .iter()
                    .flat_map(|p| p.iter())
                    .map(|&id| costs[id as usize])
                    .sum();
                let mut current_e = compute_total_e_dp(&new_id_vecs, &current_multi, &costs);
                let mut current_t = current_g + current_e;

                for (ngram, _count) in &sorted_candidates {
                    let is_dup = current_multi.iter().any(|p| p == ngram);
                    if is_dup {
                        continue;
                    }

                    let pattern_cost: f64 = ngram.iter().map(|&id| costs[id as usize]).sum();
                    let mut new_multi = current_multi.clone();
                    new_multi.push(ngram.clone());
                    let new_g = current_g + pattern_cost;
                    let new_e = compute_total_e_dp(&new_id_vecs, &new_multi, &costs);
                    let new_t = new_g + new_e;

                    if new_t < current_t {
                        let symbols: Vec<Symbol> = ngram
                            .iter()
                            .enumerate()
                            .map(|(i, &id)| {
                                let mut s = Symbol::new(id);
                                s.position = i as i32;
                                s
                            })
                            .collect();
                        let mut pat = Pattern::new(symbols, self.next_pattern_id);
                        pat.frequency = 1;
                        self.next_pattern_id += 1;
                        self.old_patterns.push(pat);
                        current_multi.push(ngram.clone());
                        current_g = new_g;
                        current_e = new_e;
                        current_t = new_t;
                    }
                }
            }

            // N-gram miner only runs cold-start (no multi-symbol grammar patterns yet).
            // Once SPMA beam extraction fires, learning is driven by beam search alone.
            let has_multi_symbol = self.old_patterns.iter().any(|p| p.symbols.len() >= 2);
            let added_this_cycle = if !has_multi_symbol {
                self.extract_frequent_ngrams(&self.new_patterns.clone(), 2)
            } else {
                false
            };

            // Recompute costs
            self.symbol_frequencies.clear();
            collect_frequencies(&mut self.symbol_frequencies, &self.old_patterns);
            collect_frequencies(&mut self.symbol_frequencies, &self.new_patterns);
            apply_symbol_costs(&self.symbol_frequencies, &mut self.old_patterns);
            apply_symbol_costs(&self.symbol_frequencies, &mut self.new_patterns);

            let old_grew = self.old_patterns.len() > old_count_before;

            // Converged when grammar didn't change this cycle: same grammar → same beam →
            // same MDL decisions → no change possible next cycle.
            if !old_grew && !added_this_cycle {
                break;
            }
            if total_cycles >= self.max_cycles {
                eprintln!(
                    "spma: learning truncated at max_cycles={} without convergence — \
                     increase SpmaEngine::max_cycles if the grammar is still growing",
                    self.max_cycles
                );
                break;
            }
        }

        // Print alignment tables for each new pattern using final grammar
        if self.verbose {
            println!("\n=== ALIGNMENT TABLES ===");
            for new_pattern in &self.new_patterns.clone() {
                if let Some(best) = self.run_recognition_cycle_beam(new_pattern) {
                    if best.cd > 0.0 || !best.old_pattern_indices.is_empty() {
                        let new_names = new_pattern.get_symbol_names(&self.interner);
                        println!(
                            "Pattern {}: {}",
                            new_pattern.pattern_id,
                            new_names.join(" ")
                        );
                        print_alignment_table(
                            new_pattern,
                            &best,
                            &self.old_patterns,
                            &self.interner,
                        );
                    }
                }
            }
        }

        // Snapshot corpus-level costs from new_patterns. Used at inference as fallback
        // for symbols not absorbed into any grammar pattern (which would otherwise cost 0).
        // Size to cover both interned atom IDs and pattern IDs (next_pattern_id).
        let max_id = self.interner.len().max(self.next_pattern_id as usize + 1);
        self.corpus_costs = vec![0.0f64; max_id];
        for p in &self.new_patterns {
            for s in &p.symbols {
                if (s.raw_id() as usize) < max_id && s.bit_cost > 0.0 {
                    self.corpus_costs[s.raw_id() as usize] = s.bit_cost;
                }
            }
        }

        // ── N-level outer loop ──────────────────────────────────────────────────
        // Save level-0 state — learn_one_level overwrites self.old_patterns/new_patterns
        let level0_old = self.old_patterns.clone();
        let level0_new = self.new_patterns.clone();

        let mut current_new = self.new_patterns.clone();
        let mut current_old = self.old_patterns.clone();
        let mut current_costs = self.corpus_costs.clone();

        loop {
            // Safety circuit-breaker — expected termination is MDL-driven (viable==0 or next_old.is_empty()).
            if self.grammar_levels.len() >= self.max_levels_safety_cap as usize {
                break;
            }

            let next_new = build_next_level_patterns(
                &current_new,
                &current_old,
                self.keep_rows as usize,
                &current_costs,
                &mut self.next_pattern_id,
            );

            // Natural termination: nothing to compress at this level.
            let viable = next_new.iter().filter(|p| p.symbols.len() >= 2).count();
            if viable == 0 {
                break;
            }

            let (next_old, next_costs) = self.learn_one_level(next_new.clone())?;

            // Natural termination: MDL gate rejected everything — no repeated structure.
            if next_old.is_empty() {
                break;
            }

            self.grammar_levels.push(GrammarLevel {
                old_patterns: next_old.clone(),
                corpus_costs: next_costs.clone(),
            });

            current_new = next_new;
            current_old = next_old;
            current_costs = next_costs;
        }

        // Restore level-0 state (inference uses self.old_patterns for level-0)
        self.old_patterns = level0_old;
        self.new_patterns = level0_new;
        // ── end N-level loop ────────────────────────────────────────────────────

        // Recompute symbol_frequencies from restored level-0 state so that
        // only atom IDs (known to the interner) are present in the map.
        self.symbol_frequencies.clear();
        collect_frequencies(&mut self.symbol_frequencies, &self.old_patterns);
        collect_frequencies(&mut self.symbol_frequencies, &self.new_patterns);

        let string_frequencies: HashMap<String, u32> = self
            .symbol_frequencies
            .iter()
            .filter_map(|(&id, &freq)| {
                if (id as usize) < self.interner.len() {
                    Some((self.interner.name(id).to_owned(), freq))
                } else {
                    None
                }
            })
            .collect();

        Ok(LearningResults {
            cycles: total_cycles,
            final_patterns: self.old_patterns.clone(),
            symbol_frequencies: string_frequencies,
            original_alphabet_size: self.original_alphabet.len(),
            final_alphabet_size: self.symbol_frequencies.len(),
            t_per_cycle: vec![],
        })
    }

    /// Extract frequent contiguous n-grams from input patterns and add to old store.
    /// Uses MDL criterion: only include a pattern if it reduces global encoding cost.
    /// Returns true if any new n-gram patterns were added.
    fn extract_frequent_ngrams(&mut self, patterns: &[Pattern], min_freq: u32) -> bool {
        let mut ngram_counts: BTreeMap<Vec<u32>, u32> = BTreeMap::new();

        for pat in patterns {
            let ids: Vec<u32> = pat.symbols.iter().map(|s| s.raw_id()).collect();
            for n in 2..=3 {
                if ids.len() >= n {
                    for window in ids.windows(n) {
                        *ngram_counts.entry(window.to_vec()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Sort candidates by (freq-1) * cost descending = net savings potential
        let max_id = self.interner.len();
        let mut costs = vec![0.0f64; max_id];
        for p in self.old_patterns.iter().chain(self.new_patterns.iter()) {
            for s in &p.symbols {
                if (s.raw_id() as usize) < max_id {
                    costs[s.raw_id() as usize] = s.bit_cost;
                }
            }
        }

        let mut candidates: Vec<(Vec<u32>, u32)> = ngram_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_freq)
            .collect();
        candidates.sort_by(|a, b| {
            let cost_a: f64 = a.0.iter().map(|&id| costs[id as usize]).sum();
            let cost_b: f64 = b.0.iter().map(|&id| costs[id as usize]).sum();
            let save_a = (a.1 as f64 - 1.0) * cost_a;
            let save_b = (b.1 as f64 - 1.0) * cost_b;
            save_b
                .partial_cmp(&save_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        // Greedy MDL selection: add each candidate only if it improves global T
        let new_id_vecs: Vec<Vec<u32>> = self
            .new_patterns
            .iter()
            .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
            .collect();

        let mut added = false;
        for (ngram, count) in &candidates {
            let is_dup = self.old_patterns.iter().any(|p| {
                let p_syms: Vec<u32> = p.symbols.iter().map(|s| s.raw_id()).collect();
                p_syms == *ngram
            });
            if is_dup || *count < min_freq {
                continue;
            }

            // Compute current global_T
            let current_multi: Vec<Vec<u32>> = self
                .old_patterns
                .iter()
                .filter(|p| p.symbols.len() >= 2)
                .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
                .collect();
            let current_g: f64 = self
                .old_patterns
                .iter()
                .filter(|p| p.symbols.len() >= 2)
                .flat_map(|p| p.symbols.iter())
                .map(|s| costs[s.raw_id() as usize])
                .sum();
            let current_e = compute_total_e_dp(&new_id_vecs, &current_multi, &costs);
            let current_t = current_g + current_e;

            // Compute global_T with this pattern added
            let pattern_cost: f64 = ngram.iter().map(|&id| costs[id as usize]).sum();
            let mut new_multi = current_multi.clone();
            new_multi.push(ngram.clone());
            let new_g = current_g + pattern_cost;
            let new_e = compute_total_e_dp(&new_id_vecs, &new_multi, &costs);
            let new_t = new_g + new_e;

            // Only add if it improves (decreases) global T
            if new_t < current_t {
                let symbols: Vec<Symbol> = ngram
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| {
                        let mut s = Symbol::new(id);
                        s.position = i as i32;
                        s
                    })
                    .collect();
                let mut pat = Pattern::new(symbols, self.next_pattern_id);
                pat.frequency = *count;
                self.next_pattern_id += 1;
                self.old_patterns.push(pat);
                added = true;
            }
        }

        added
    }

    fn extract_frequent_ngrams_ids(&mut self, patterns: &[Pattern], min_freq: u32) -> bool {
        let mut ngram_counts: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
        for pat in patterns {
            let ids: Vec<u32> = pat.symbols.iter().map(|s| s.raw_id()).collect();
            if ids.len() >= 2 {
                for window in ids.windows(2) {
                    *ngram_counts.entry(window.to_vec()).or_insert(0) += 1;
                }
            }
        }

        let max_id = self.next_pattern_id as usize;
        let mut costs = vec![0.0f64; max_id.max(1)];
        for p in self.old_patterns.iter().chain(self.new_patterns.iter()) {
            for s in &p.symbols {
                let id = s.raw_id() as usize;
                if id < costs.len() {
                    costs[id] = s.bit_cost;
                }
            }
        }

        let mut candidates: Vec<(Vec<u32>, u32)> = ngram_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_freq)
            .collect();
        candidates.sort_by(|a, b| {
            let cost_a: f64 = a.0.iter().map(|&id| costs.get(id as usize).copied().unwrap_or(0.0)).sum();
            let cost_b: f64 = b.0.iter().map(|&id| costs.get(id as usize).copied().unwrap_or(0.0)).sum();
            let save_a = (a.1 as f64 - 1.0) * cost_a;
            let save_b = (b.1 as f64 - 1.0) * cost_b;
            save_b.partial_cmp(&save_a).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let new_id_vecs: Vec<Vec<u32>> = self
            .new_patterns
            .iter()
            .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
            .collect();

        let mut added = false;
        for (ngram, count) in &candidates {
            let is_dup = self.old_patterns.iter().any(|p| {
                let p_ids: Vec<u32> = p.symbols.iter().map(|s| s.raw_id()).collect();
                p_ids == *ngram
            });
            if is_dup || *count < min_freq {
                continue;
            }

            let current_multi: Vec<Vec<u32>> = self
                .old_patterns
                .iter()
                .filter(|p| p.symbols.len() >= 2)
                .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
                .collect();
            let current_g: f64 = self
                .old_patterns
                .iter()
                .filter(|p| p.symbols.len() >= 2)
                .flat_map(|p| p.symbols.iter())
                .map(|s| costs.get(s.raw_id() as usize).copied().unwrap_or(0.0))
                .sum();
            let current_e = compute_total_e_dp(&new_id_vecs, &current_multi, &costs);
            let current_t = current_g + current_e;

            let pattern_cost: f64 = ngram.iter()
                .map(|&id| costs.get(id as usize).copied().unwrap_or(0.0))
                .sum();
            let mut new_multi = current_multi.clone();
            new_multi.push(ngram.clone());
            let new_g = current_g + pattern_cost;
            let new_e = compute_total_e_dp(&new_id_vecs, &new_multi, &costs);
            let new_t = new_g + new_e;

            if new_t < current_t {
                let symbols: Vec<Symbol> = ngram
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| {
                        let mut s = Symbol::new_pattern_ref(id);
                        s.position = i as i32;
                        s
                    })
                    .collect();
                let mut pat = Pattern::new(symbols, self.next_pattern_id);
                pat.frequency = *count;
                self.next_pattern_id += 1;
                self.old_patterns.push(pat);
                added = true;
            }
        }

        added
    }

    fn learn_one_level(&mut self, new_pats: Vec<Pattern>) -> Result<(Vec<Pattern>, Vec<f64>)> {
        self.new_patterns = new_pats;
        self.old_patterns.clear();

        self.symbol_frequencies.clear();
        collect_frequencies(&mut self.symbol_frequencies, &self.new_patterns);
        apply_symbol_costs(&self.symbol_frequencies, &mut self.new_patterns);

        let mut total_cycles = 0u32;

        loop {
            total_cycles += 1;
            let old_count_before = self.old_patterns.len();

            let new_patterns_snapshot = self.new_patterns.clone();
            let mut candidates: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
            for new_pattern in &new_patterns_snapshot {
                let best_opt = self.run_recognition_cycle_beam(new_pattern);
                if let Some(best) = best_opt {
                    if best.cd > 0.0 {
                        for &oi in &best.old_pattern_indices {
                            if oi < self.old_patterns.len() {
                                self.old_patterns[oi].frequency += 1;
                            }
                        }
                        for learned in extract_learned_patterns(
                            new_pattern,
                            &best.covered_new,
                            &mut self.next_pattern_id,
                        ) {
                            let learned_ids: Vec<u32> =
                                learned.symbols.iter().map(|s| s.raw_id()).collect();
                            *candidates.entry(learned_ids).or_insert(0) += 1;
                        }
                    }
                }
            }

            {
                let max_id = self.next_pattern_id as usize;
                let mut costs = vec![0.0f64; max_id.max(1)];
                for p in self.old_patterns.iter().chain(self.new_patterns.iter()) {
                    for s in &p.symbols {
                        let id = s.raw_id() as usize;
                        if id < costs.len() {
                            costs[id] = s.bit_cost;
                        }
                    }
                }
                let new_id_vecs: Vec<Vec<u32>> = self
                    .new_patterns
                    .iter()
                    .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
                    .collect();

                let mut sorted_candidates: Vec<(Vec<u32>, u32)> = candidates.into_iter().collect();
                sorted_candidates.sort_by(|a, b| {
                    let cost_a: f64 = a.0.iter().map(|&id| costs.get(id as usize).copied().unwrap_or(0.0)).sum();
                    let cost_b: f64 = b.0.iter().map(|&id| costs.get(id as usize).copied().unwrap_or(0.0)).sum();
                    let save_a = (a.1 as f64 - 1.0) * cost_a;
                    let save_b = (b.1 as f64 - 1.0) * cost_b;
                    save_b.partial_cmp(&save_a).unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });

                let mut current_multi: Vec<Vec<u32>> = self
                    .old_patterns
                    .iter()
                    .filter(|p| p.symbols.len() >= 2)
                    .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
                    .collect();
                let mut current_g: f64 = current_multi
                    .iter()
                    .flat_map(|p| p.iter())
                    .map(|&id| costs.get(id as usize).copied().unwrap_or(0.0))
                    .sum();
                let mut current_e = compute_total_e_dp(&new_id_vecs, &current_multi, &costs);
                let mut current_t = current_g + current_e;

                for (ngram, _count) in &sorted_candidates {
                    let is_dup = current_multi.iter().any(|p| p == ngram);
                    if is_dup { continue; }

                    let pattern_cost: f64 = ngram.iter()
                        .map(|&id| costs.get(id as usize).copied().unwrap_or(0.0))
                        .sum();
                    let mut new_multi = current_multi.clone();
                    new_multi.push(ngram.clone());
                    let new_g = current_g + pattern_cost;
                    let new_e = compute_total_e_dp(&new_id_vecs, &new_multi, &costs);
                    let new_t = new_g + new_e;

                    if new_t < current_t {
                        let symbols: Vec<Symbol> = ngram
                            .iter()
                            .enumerate()
                            .map(|(i, &id)| {
                                let mut s = Symbol::new_pattern_ref(id);
                                s.position = i as i32;
                                s
                            })
                            .collect();
                        let mut pat = Pattern::new(symbols, self.next_pattern_id);
                        pat.frequency = 1;
                        self.next_pattern_id += 1;
                        self.old_patterns.push(pat);
                        current_multi.push(ngram.clone());
                        current_g = new_g;
                        current_e = new_e;
                        current_t = new_t;
                    }
                }
            }

            let has_multi_symbol = self.old_patterns.iter().any(|p| p.symbols.len() >= 2);
            let added_this_cycle = if !has_multi_symbol {
                self.extract_frequent_ngrams_ids(&self.new_patterns.clone(), 2)
            } else {
                false
            };

            self.symbol_frequencies.clear();
            collect_frequencies(&mut self.symbol_frequencies, &self.old_patterns);
            collect_frequencies(&mut self.symbol_frequencies, &self.new_patterns);
            apply_symbol_costs(&self.symbol_frequencies, &mut self.old_patterns);
            apply_symbol_costs(&self.symbol_frequencies, &mut self.new_patterns);

            let old_grew = self.old_patterns.len() > old_count_before;
            if !old_grew && !added_this_cycle {
                break;
            }
            if total_cycles >= self.max_cycles {
                eprintln!(
                    "spma: learn_one_level truncated at max_cycles={} without convergence",
                    self.max_cycles
                );
                break;
            }
        }

        // Snapshot corpus costs for this level
        let max_id = self.next_pattern_id as usize;
        let mut level_corpus_costs = vec![0.0f64; max_id.max(1)];
        for p in &self.new_patterns {
            for s in &p.symbols {
                let id = s.raw_id() as usize;
                if id < level_corpus_costs.len() && s.bit_cost > 0.0 {
                    level_corpus_costs[id] = s.bit_cost;
                }
            }
        }

        let old_patterns = self.old_patterns.clone();
        Ok((old_patterns, level_corpus_costs))
    }

    /// Compute compression ratio using global MDL formula:
    /// - global_G = cost of storing grammar (all old patterns, each counted once)
    /// - global_E = sum of uncovered symbol costs across all new patterns
    /// - compression_ratio = total_raw / (global_G + global_E)
    pub fn compute_global_compression_ratio(
        &self,
        new_patterns: &[Pattern],
        old_patterns: &[Pattern],
        _beam_k: usize,
    ) -> f64 {
        let max_id = self.interner.len();
        let mut costs = vec![0.0f64; max_id];
        for p in old_patterns.iter().chain(new_patterns.iter()) {
            for s in &p.symbols {
                if (s.raw_id() as usize) < max_id {
                    costs[s.raw_id() as usize] = s.bit_cost;
                }
            }
        }

        // Global G: cost of storing the grammar (multi-symbol patterns only).
        // Single-symbol seeds are the alphabet baseline, not grammar overhead.
        let global_g: f64 = old_patterns
            .iter()
            .filter(|p| p.symbols.len() >= 2)
            .flat_map(|p| p.symbols.iter())
            .map(|s| costs[s.raw_id() as usize])
            .sum();

        // Collect multi-symbol grammar patterns as ID sequences
        let multi_id_vecs: Vec<Vec<u32>> = old_patterns
            .iter()
            .filter(|p| p.symbols.len() >= 2)
            .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
            .collect();

        let mut total_e = 0.0f64;
        let mut total_raw = 0.0f64;

        for new_pat in new_patterns {
            let new_ids: Vec<u32> = new_pat.symbols.iter().map(|s| s.raw_id()).collect();
            let raw_cost: f64 = new_ids.iter().map(|&id| costs[id as usize]).sum();
            total_raw += raw_cost;

            // Find all possible pattern matches (position, length)
            let mut matches: Vec<(usize, usize)> = Vec::new();
            for pat_ids in &multi_id_vecs {
                if pat_ids.len() > new_ids.len() {
                    continue;
                }
                for start in 0..=(new_ids.len() - pat_ids.len()) {
                    if new_ids[start..start + pat_ids.len()] == pat_ids[..] {
                        matches.push((start, pat_ids.len()));
                    }
                }
            }

            // Find optimal non-overlapping coverage via DP
            // dp[i] = min uncovered cost for positions [i..n]
            let n = new_ids.len();
            let mut dp = vec![0.0f64; n + 1];
            for i in (0..n).rev() {
                // Option 1: leave position i uncovered
                dp[i] = costs[new_ids[i] as usize] + dp[i + 1];
                // Option 2: use a pattern starting at i
                for &(start, len) in &matches {
                    if start == i {
                        let after = i + len;
                        if after <= n && dp[after] < dp[i] {
                            dp[i] = dp[after];
                        }
                    }
                }
            }

            total_e += dp[0];
        }

        let global_t = global_g + total_e;
        if global_t > 0.0 {
            total_raw / global_t
        } else {
            1.0
        }
    }
}

/// Extract all maximal contiguous covered spans from a New pattern as candidate grammar patterns.
/// Spans shorter than 2 symbols are discarded. Each span gets a unique id from `next_id`.
pub fn extract_learned_patterns(
    new_pattern: &Pattern,
    covered: &[bool],
    next_id: &mut u32,
) -> Vec<Pattern> {
    let symbols = &new_pattern.symbols;
    let n = symbols.len();
    let mut result = Vec::new();
    let mut i = 0;
    while i < n {
        if covered[i] {
            let start = i;
            while i < n && covered[i] {
                i += 1;
            }
            if i - start >= 2 {
                let span = symbols[start..i].to_vec();
                result.push(Pattern::new(span, *next_id));
                *next_id += 1;
            }
        } else {
            i += 1;
        }
    }
    result
}

fn build_next_level_patterns(
    new_pats: &[Pattern],
    old_pats: &[Pattern],
    keep_rows: usize,
    costs: &[f64],
    next_id: &mut u32,
) -> Vec<Pattern> {
    let old_id_vecs: Vec<Vec<u32>> = old_pats
        .iter()
        .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
        .collect();

    new_pats
        .iter()
        .filter_map(|np| {
            let ids: Vec<u32> = np.symbols.iter().map(|s| s.raw_id()).collect();
            if ids.is_empty() {
                return None;
            }

            let best = beam_search(&ids, &old_id_vecs, keep_rows, costs)
                .into_iter()
                .next()?;

            // Extract used pattern IDs ordered by first covered New position
            let mut pid_starts: Vec<(u32, usize)> = best
                .old_pattern_indices
                .iter()
                .filter_map(|&oi| {
                    let pid = old_pats[oi].pattern_id;
                    let start = ids
                        .iter()
                        .enumerate()
                        .find(|&(i, &sym_id)| {
                            best.covered_new[i] && old_id_vecs[oi].contains(&sym_id)
                        })
                        .map(|(i, _)| i)?;
                    Some((pid, start))
                })
                .collect();
            pid_starts.sort_by_key(|&(_, s)| s);
            let pid_seq: Vec<u32> = pid_starts.into_iter().map(|(pid, _)| pid).collect();

            if pid_seq.is_empty() {
                return None;
            }

            let symbols: Vec<Symbol> = pid_seq
                .iter()
                .map(|&pid| Symbol::new_pattern_ref(pid))
                .collect();
            let pat = Pattern::new(symbols, *next_id);
            *next_id += 1;
            Some(pat)
        })
        .collect()
}

/// Compute total E (uncovered position costs) using DP optimal tiling.
fn compute_total_e_dp(sentences: &[Vec<u32>], grammar: &[Vec<u32>], costs: &[f64]) -> f64 {
    let mut total_e = 0.0;
    for sent in sentences {
        let n = sent.len();
        // dp[i] = min uncovered cost for positions [i..n]
        let mut dp = vec![0.0f64; n + 1];
        for i in (0..n).rev() {
            // Option 1: leave position i uncovered
            dp[i] = costs[sent[i] as usize] + dp[i + 1];
            // Option 2: use a grammar pattern starting at i
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


#[cfg(test)]
mod tests {
    use super::*;

    fn sym(id: u32) -> Symbol {
        Symbol::new(id)
    }

    fn pat(ids: &[u32]) -> Pattern {
        Pattern::new(ids.iter().map(|&id| sym(id)).collect(), 0)
    }

    #[test]
    fn contiguous_full_coverage() {
        // covered = [T,T,T] → one span [0,1,2]
        let p = pat(&[0, 1, 2]);
        let mut id = 10u32;
        let spans = extract_learned_patterns(&p, &[true, true, true], &mut id);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].symbols.iter().map(|s| s.raw_id()).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(id, 11);
    }

    #[test]
    fn gap_produces_two_spans() {
        // covered = [T,T,F,T,T] → spans [0,1] and [3,4]
        let p = pat(&[0, 1, 2, 3, 4]);
        let mut id = 1u32;
        let spans = extract_learned_patterns(&p, &[true, true, false, true, true], &mut id);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].symbols.iter().map(|s| s.raw_id()).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(spans[1].symbols.iter().map(|s| s.raw_id()).collect::<Vec<_>>(), vec![3, 4]);
        assert_eq!(id, 3);
    }

    #[test]
    fn single_symbol_spans_discarded() {
        // covered = [T,F,T] → both spans length 1, nothing returned
        let p = pat(&[0, 1, 2]);
        let mut id = 5u32;
        let spans = extract_learned_patterns(&p, &[true, false, true], &mut id);
        assert!(spans.is_empty());
        assert_eq!(id, 5, "id counter must not advance for discarded spans");
    }

    #[test]
    fn no_coverage_returns_empty() {
        let p = pat(&[0, 1, 2]);
        let mut id = 1u32;
        let spans = extract_learned_patterns(&p, &[false, false, false], &mut id);
        assert!(spans.is_empty());
        assert_eq!(id, 1);
    }

    #[test]
    fn mixed_gap_at_start_and_end() {
        // covered = [F,T,T,T,F] → one span [1,2,3]
        let p = pat(&[0, 1, 2, 3, 4]);
        let mut id = 7u32;
        let spans = extract_learned_patterns(&p, &[false, true, true, true, false], &mut id);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].symbols.iter().map(|s| s.raw_id()).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(id, 8);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningResults {
    pub cycles: u32,
    pub final_patterns: Vec<Pattern>,
    pub symbol_frequencies: HashMap<String, u32>,
    pub original_alphabet_size: usize,
    pub final_alphabet_size: usize,
    /// T (total encoding cost) at the end of each cycle — should be monotonically non-increasing.
    pub t_per_cycle: Vec<f64>,
}
