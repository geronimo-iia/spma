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
pub use engine::{extract_learned_patterns, write_alignment_table, LearningResults, SpmaEngine};

// ── Public API ────────────────────────────────────────────────────────────────

/// Result of inferring a single sequence against the learned grammar.
#[derive(Debug, Clone)]
pub struct InferResult {
    /// Encoding cost of symbols not covered by the grammar (E in T=G+E).
    pub e_cost: f64,
    /// Compression difference: positive means the grammar compresses the sequence.
    pub cd: f64,
    /// True when E > 0 (unmatched symbols remain).
    pub is_anomaly: bool,
    /// Symbol names not covered by any grammar pattern.
    pub unmatched: Vec<String>,
    /// Human-readable alignment table.
    pub alignment: String,
}

/// Serializable snapshot of the learned grammar for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GrammarSnapshot {
    old_patterns: Vec<Pattern>,
    interner_names: Vec<String>,
    corpus_costs: Vec<f64>,
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
        let snapshot = GrammarSnapshot {
            old_patterns: self.inner.old_patterns.clone(),
            interner_names: (0..self.inner.interner.len())
                .map(|i| self.inner.interner.name(i as u32).to_owned())
                .collect(),
            corpus_costs: self.inner.corpus_costs.clone(),
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
            e_cost,
            cd,
            is_anomaly: e_cost > 0.0,
            unmatched,
            alignment: alignment_str,
        })
    }
}

