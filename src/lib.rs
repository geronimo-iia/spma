//! SPMA — SP Multiple Alignment
//!
//! Symbolic sequential anomaly detection via T=G+E scoring.
//! Learns grammars from discrete event sequences; detects anomalies when E > 0.
//!
//! # Quick Start
//!
//! ```no_run
//! use spma::Spma;
//!
//! let mut engine = Spma::new();
//! engine.train(&[
//!     vec!["fault_A", "fault_B", "fault_C"],
//!     vec!["fault_A", "fault_B", "fault_D"],
//! ]).unwrap();
//! engine.save("spma_grammar.bin").unwrap();
//!
//! let engine = Spma::load("spma_grammar.bin").unwrap();
//! let result = engine.infer(&["fault_A", "fault_B", "fault_C"]).unwrap();
//! println!("E={:.2}  CD={:+.2}  anomaly={}", result.e_cost, result.cd, result.is_anomaly);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod intern;
pub use intern::Interner;

pub(crate) mod model;
pub use model::{compute_t_ge, format_symbol};
pub use model::{AlignmentType, Pattern, Symbol, SymbolStatus, SymbolType};

pub(crate) mod beam;
pub use beam::{beam_search, BeamAlignment};

pub(crate) mod engine;
pub use engine::{extract_learned_patterns, write_alignment_table, GrammarLevel, LearningResults, SpmaEngine};
pub use model::SymbolRef;

// ── Public API ────────────────────────────────────────────────────────────────

/// Result of inferring a single sequence against the learned grammar.
#[derive(Debug, Clone)]
pub struct InferResult {
    /// Total encoding cost across all levels: e_cost_l0 + sum(level_costs).
    pub e_cost: f64,
    /// Compression difference: positive means the grammar compresses the sequence.
    pub cd: f64,
    /// True when E > 0 (unmatched symbols remain).
    pub is_anomaly: bool,
    /// Symbol names not covered by any grammar pattern.
    pub unmatched: Vec<String>,
    /// Human-readable alignment table.
    pub alignment: String,
    /// E cost per grammar level beyond 0: [level-1, level-2, ...]. Empty when only level-0 exists.
    pub level_costs: Vec<f64>,
    /// Alignment string per grammar level beyond 0.
    pub level_alignments: Vec<String>,
}

/// Serializable snapshot of a single grammar level for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GrammarLevelSnapshot {
    old_patterns: Vec<Pattern>,
    corpus_costs: Vec<f64>,
}

/// Serializable snapshot of the learned grammar for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GrammarSnapshot {
    format_version: u8,
    old_patterns: Vec<Pattern>,
    interner_names: Vec<String>,
    corpus_costs: Vec<f64>,
    grammar_levels: Vec<GrammarLevelSnapshot>,
}

/// Primary entry point for library users.
pub struct Spma {
    inner: SpmaEngine,
}

impl Default for Spma {
    fn default() -> Self {
        Self::new()
    }
}

impl Spma {
    pub fn new() -> Self {
        Self {
            inner: SpmaEngine::new(),
        }
    }

    pub fn set_max_cycles(&mut self, n: u32) {
        self.inner.max_cycles = n;
    }

    pub fn grammar_size(&self) -> usize {
        self.inner.old_patterns.iter().filter(|p| p.symbols.len() >= 2).count()
    }

    pub fn grammar_depth(&self) -> usize {
        1 + self.inner.grammar_levels.len()
    }

    pub fn grammar_size_at(&self, level: usize) -> usize {
        if level == 0 {
            self.inner.old_patterns.iter().filter(|p| p.symbols.len() >= 2).count()
        } else {
            self.inner
                .grammar_levels
                .get(level - 1)
                .map(|gl| gl.old_patterns.iter().filter(|p| p.symbols.len() >= 2).count())
                .unwrap_or(0)
        }
    }

    pub fn set_max_levels_safety_cap(&mut self, n: u8) {
        self.inner.max_levels_safety_cap = n;
    }

    /// Train on a corpus of sequences. Each sequence is a slice of symbol name strings.
    pub fn train(&mut self, sequences: &[Vec<&str>]) -> Result<()> {
        let mut patterns = Vec::new();
        for (i, seq) in sequences.iter().enumerate() {
            let line = seq.join(" ");
            let mut symbols = Vec::new();
            for (pos, name) in seq.iter().enumerate() {
                use crate::{SymbolStatus, SymbolType};
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
                let id = self.inner.interner.intern(canonical_name);
                let mut sym = Symbol::new(id);
                sym.position = pos as i32;
                sym.symbol_type = sym_type;
                sym.status = sym_status;
                symbols.push(sym);
                self.inner.original_alphabet.insert(id);
            }
            if !symbols.is_empty() {
                let pat = Pattern::new(symbols, self.inner.next_pattern_id);
                self.inner.next_pattern_id += 1;
                patterns.push(pat);
            }
            let _ = line;
            let _ = i;
        }
        self.inner.learn(patterns)?;
        Ok(())
    }

    /// Save learned grammar to a binary file.
    pub fn save(&self, path: &str) -> Result<()> {
        let grammar_levels: Vec<GrammarLevelSnapshot> = self
            .inner
            .grammar_levels
            .iter()
            .map(|gl| GrammarLevelSnapshot {
                old_patterns: gl.old_patterns.clone(),
                corpus_costs: gl.corpus_costs.clone(),
            })
            .collect();
        let snapshot = GrammarSnapshot {
            format_version: 2,
            old_patterns: self.inner.old_patterns.clone(),
            interner_names: (0..self.inner.interner.len())
                .map(|i| self.inner.interner.name(i as u32).to_owned())
                .collect(),
            corpus_costs: self.inner.corpus_costs.clone(),
            grammar_levels,
        };
        let bytes =
            bincode::serialize(&snapshot).map_err(|e| anyhow::anyhow!("bincode serialize: {e}"))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a previously saved grammar from a binary file.
    pub fn load(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let snapshot: GrammarSnapshot = bincode::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("bincode deserialize: {e}"))?;
        let mut engine = Self::new();
        for name in &snapshot.interner_names {
            engine.inner.interner.intern(name);
        }
        engine.inner.old_patterns = snapshot.old_patterns;
        engine.inner.corpus_costs = snapshot.corpus_costs;
        engine.inner.grammar_levels = snapshot.grammar_levels.into_iter().map(|gl| {
            crate::engine::GrammarLevel {
                old_patterns: gl.old_patterns,
                corpus_costs: gl.corpus_costs,
            }
        }).collect();
        // All interned names were seen during training → all belong to original_alphabet.
        for i in 0..engine.inner.interner.len() {
            engine.inner.original_alphabet.insert(i as u32);
        }
        Ok(engine)
    }

    /// Infer a single sequence against the learned grammar. Does not modify state.
    pub fn infer(&self, sequence: &[&str]) -> Result<InferResult> {
        let mut tmp_interner = self.inner.interner.clone();
        let ids: Vec<u32> = sequence.iter().map(|&s| tmp_interner.intern(s)).collect();

        // Identify positions with symbols not seen during training.
        let unknown: Vec<bool> = ids
            .iter()
            .map(|id| !self.inner.original_alphabet.contains(id))
            .collect();

        // Average bit cost of known symbols — used as penalty for unknown ones.
        let known_costs: Vec<f64> = self
            .inner
            .old_patterns
            .iter()
            .flat_map(|p| p.symbols.iter())
            .map(|s| s.bit_cost)
            .filter(|&c| c > 0.0)
            .collect();
        let unknown_penalty = if known_costs.is_empty() {
            1.0
        } else {
            known_costs.iter().sum::<f64>() / known_costs.len() as f64
        };

        let max_id = tmp_interner.len();
        let mut costs = vec![0.0f64; max_id];
        for p in &self.inner.old_patterns {
            for s in &p.symbols {
                if (s.raw_id() as usize) < max_id {
                    costs[s.raw_id() as usize] = s.bit_cost;
                }
            }
        }
        // Fall back to corpus costs for symbols present during training but not in any
        // grammar pattern (which would otherwise cost 0, masking uncovered symbols).
        for (id, &cc) in self.inner.corpus_costs.iter().enumerate() {
            if id < max_id && costs[id] == 0.0 && cc > 0.0 {
                costs[id] = cc;
            }
        }

        let old_id_vecs: Vec<Vec<u32>> = self
            .inner
            .old_patterns
            .iter()
            .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
            .collect();

        let best_opt = beam_search(&ids, &old_id_vecs, self.inner.keep_rows as usize, &costs)
            .into_iter()
            .next();

        let (beam_e, beam_cd, mut covered) = if let Some(ref b) = best_opt {
            (b.e, b.cd, b.covered_new.clone())
        } else {
            let raw: f64 = ids
                .iter()
                .map(|&id| {
                    if (id as usize) < costs.len() {
                        costs[id as usize]
                    } else {
                        0.0
                    }
                })
                .sum();
            (raw, 0.0, vec![false; ids.len()])
        };

        // Unknown positions are always uncovered regardless of beam result.
        let mut unknown_e = 0.0f64;
        for (i, &is_unknown) in unknown.iter().enumerate() {
            if is_unknown {
                covered[i] = false;
                unknown_e += unknown_penalty;
            }
        }
        let e_cost = beam_e + unknown_e;
        // CD is reduced by the penalty we're adding for unknown symbols.
        let cd = beam_cd - unknown_e;

        // N-level inference loop
        let mut level_costs: Vec<f64> = Vec::new();
        let mut level_alignments: Vec<String> = Vec::new();

        // prev_alignment holds the BeamAlignment from the previous level.
        // For level-1, that is the level-0 beam alignment.
        let mut prev_alignment_opt: Option<BeamAlignment> = best_opt.clone();
        let mut prev_old_patterns_snapshot: Vec<Pattern> = self.inner.old_patterns.clone();
        // prev_seq tracks the sequence fed into the previous level's beam search.
        // Starts as the original atom IDs; becomes pid_seq after each level.
        let mut prev_seq: Vec<u32> = ids.clone();

        for level_idx in 0..self.inner.grammar_levels.len() {
            let Some(ref prev_align) = prev_alignment_opt else { break };

            let level = &self.inner.grammar_levels[level_idx];

            // Build pid_sequence from the previous level's alignment.
            let prev_old_id_vecs: Vec<Vec<u32>> = prev_old_patterns_snapshot
                .iter()
                .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
                .collect();

            let mut pid_starts: Vec<(u32, usize)> = prev_align
                .old_pattern_indices
                .iter()
                .filter_map(|&oi| {
                    let pid = prev_old_patterns_snapshot[oi].pattern_id;
                    // Find the first covered position in prev_seq that this old pattern contributed to.
                    let start = prev_seq
                        .iter()
                        .enumerate()
                        .find(|&(i, &sym_id)| {
                            prev_align.covered_new[i] && prev_old_id_vecs[oi].contains(&sym_id)
                        })
                        .map(|(i, _)| i)?;
                    Some((pid, start))
                })
                .collect();
            pid_starts.sort_by_key(|&(_, s)| s);
            let pid_seq: Vec<u32> = pid_starts.into_iter().map(|(pid, _)| pid).collect();

            if pid_seq.is_empty() {
                break;
            }

            // Build cost table for this level.
            let max_pid = level.corpus_costs.len().max(
                level.old_patterns.iter()
                    .flat_map(|p| p.symbols.iter())
                    .map(|s| s.raw_id() as usize + 1)
                    .max()
                    .unwrap_or(1)
            );
            let mut level_cost_table = vec![0.0f64; max_pid];
            for p in &level.old_patterns {
                for s in &p.symbols {
                    let id = s.raw_id() as usize;
                    if id < level_cost_table.len() {
                        level_cost_table[id] = s.bit_cost;
                    }
                }
            }
            for (id, &cc) in level.corpus_costs.iter().enumerate() {
                if id < level_cost_table.len() && level_cost_table[id] == 0.0 && cc > 0.0 {
                    level_cost_table[id] = cc;
                }
            }

            let level_old_id_vecs: Vec<Vec<u32>> = level
                .old_patterns
                .iter()
                .map(|p| p.symbols.iter().map(|s| s.raw_id()).collect())
                .collect();

            // pid_seq IDs must be within level_cost_table bounds.
            let needed = pid_seq.iter().map(|&id| id as usize + 1).max().unwrap_or(0);
            let cost_table = if needed > level_cost_table.len() {
                let mut t = level_cost_table.clone();
                t.resize(needed, 0.0);
                t
            } else {
                level_cost_table
            };

            let level_align_opt = beam_search(
                &pid_seq,
                &level_old_id_vecs,
                self.inner.keep_rows as usize,
                &cost_table,
            )
            .into_iter()
            .next();

            let level_e = if let Some(ref la) = level_align_opt {
                la.e
            } else {
                pid_seq
                    .iter()
                    .map(|&id| cost_table.get(id as usize).copied().unwrap_or(0.0))
                    .sum()
            };

            level_costs.push(level_e);

            // Build level alignment string using pid → display_name map.
            let level_align_str = {
                use std::fmt::Write as FmtWrite;
                let pid_to_name: std::collections::HashMap<u32, String> = level
                    .old_patterns
                    .iter()
                    .map(|p| {
                        let name = p
                            .symbols
                            .iter()
                            .map(|s| match s.name {
                                crate::model::SymbolRef::Atom(id) => {
                                    format!("[pat:{}]", id)
                                }
                                crate::model::SymbolRef::Pattern(pid) => format!("[pat:{}]", pid),
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        (p.pattern_id, format!("[{}]", name))
                    })
                    .collect();

                let mut out = String::new();
                let pid_names: Vec<String> = pid_seq
                    .iter()
                    .map(|&pid| pid_to_name.get(&pid).cloned().unwrap_or(format!("[pid:{}]", pid)))
                    .collect();

                let _ = writeln!(out, "Level-{} alignment:", level_idx + 1);
                let _ = writeln!(out, "New: {}", pid_names.join("  "));
                if let Some(ref la) = level_align_opt {
                    for &oi in &la.old_pattern_indices {
                        if let Some(op) = level.old_patterns.get(oi) {
                            let op_name = pid_to_name.get(&op.pattern_id).cloned().unwrap_or_default();
                            let _ = writeln!(out, "Old: {}", op_name);
                        }
                    }
                }
                let _ = writeln!(out, "E={:.1} bits", level_e);
                out
            };

            level_alignments.push(level_align_str);
            prev_alignment_opt = level_align_opt;
            prev_old_patterns_snapshot = level.old_patterns.clone();
            prev_seq = pid_seq;
        }

        let total_e_cost = e_cost + level_costs.iter().sum::<f64>();

        let unmatched: Vec<String> = ids
            .iter()
            .enumerate()
            .filter(|&(i, _)| !covered[i])
            .map(|(_, &id)| tmp_interner.name(id).to_owned())
            .collect();

        let alignment_str = if let Some(ref best) = best_opt {
            let new_syms: Vec<Symbol> = ids
                .iter()
                .enumerate()
                .map(|(pos, &id)| {
                    let mut s = Symbol::new(id);
                    s.position = pos as i32;
                    s
                })
                .collect();
            let new_pat = Pattern::new(new_syms, 0);

            let mut out = String::new();
            write_alignment_table(&mut out, &new_pat, best, &self.inner.old_patterns, &tmp_interner);
            out
        } else {
            format!("New:  {}\n(no alignment)\n", sequence.join("  "))
        };

        Ok(InferResult {
            e_cost: total_e_cost,
            cd,
            is_anomaly: total_e_cost > 0.0,
            unmatched,
            alignment: alignment_str,
            level_costs,
            level_alignments,
        })
    }
}

