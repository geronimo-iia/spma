use serde::{Deserialize, Serialize};

use crate::Interner;

// ── SymbolRef ─────────────────────────────────────────────────────────────────

/// A reference to either an interned atom or a learned pattern.
///
/// Two variants only — no Gap (gaps are `Pattern::gaps`), no IC/OC, no variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolRef {
    /// Interned string ID from `Grammar::interner`.
    Atom(u32),
    /// ID of a learned pattern at a lower grammar level.
    Pattern(u32),
}

// ── GapConstraint ─────────────────────────────────────────────────────────────

/// Constraint between two adjacent symbols in a pattern.
///
/// Stored as a parallel vec `Pattern::gaps` alongside `Pattern::symbols`.
/// `gaps[i]` constrains the number of New positions skipped between
/// `symbols[i]` and `symbols[i+1]`.
///
/// `gaps` is empty for contiguous patterns (common case).
/// When non-empty, `gaps.len() == symbols.len() - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GapConstraint {
    /// Minimum New positions to skip (0 = adjacent is OK).
    pub min: usize,
    /// Maximum New positions to skip (`usize::MAX` = unbounded).
    pub max: usize,
}

impl GapConstraint {
    pub fn new(min: usize, max: usize) -> Self {
        Self { min, max }
    }

    /// Adjacent symbols — no gap allowed.
    pub fn none() -> Self {
        Self { min: 0, max: 0 }
    }

    /// Any gap up to `max` positions.
    pub fn up_to(max: usize) -> Self {
        Self { min: 0, max }
    }
}

// ── Pattern ───────────────────────────────────────────────────────────────────

/// A learned grammar pattern.
///
/// `symbols` contains OC content symbols only (`SymbolRef::Atom` or `SymbolRef::Pattern`).
/// `gaps` is a parallel vec of constraints between adjacent symbols.
///
/// Invariant: `gaps.is_empty() || gaps.len() == symbols.len() - 1`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: u32,
    pub symbols: Vec<SymbolRef>,
    /// Empty = contiguous pattern. `len == symbols.len() - 1` when non-contiguous.
    pub gaps: Vec<GapConstraint>,
    /// Match count accumulated during training.
    pub frequency: u32,
    /// Grammar level this pattern was induced at (0 = atom level).
    pub level: u8,
}

impl Pattern {
    /// Contiguous pattern — no gaps between symbols.
    pub fn new_contiguous(id: u32, symbols: Vec<SymbolRef>, level: u8) -> Self {
        Self {
            id,
            symbols,
            gaps: Vec::new(),
            frequency: 1,
            level,
        }
    }

    /// Non-contiguous pattern with gap constraints between adjacent symbols.
    ///
    /// # Panics (debug only)
    /// `gaps.len()` must equal `symbols.len() - 1`.
    pub fn new_with_gaps(
        id: u32,
        symbols: Vec<SymbolRef>,
        gaps: Vec<GapConstraint>,
        level: u8,
    ) -> Self {
        debug_assert!(
            gaps.is_empty() || gaps.len() == symbols.len().saturating_sub(1),
            "gaps.len() must be symbols.len()-1 or 0, got gaps={} symbols={}",
            gaps.len(),
            symbols.len()
        );
        Self {
            id,
            symbols,
            gaps,
            frequency: 1,
            level,
        }
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// True if this pattern has no gap constraints (all symbols are contiguous).
    pub fn is_contiguous(&self) -> bool {
        self.gaps.is_empty()
    }

    /// Gap constraint between `symbols[i]` and `symbols[i+1]`.
    /// Returns `None` for contiguous patterns or out-of-range index.
    pub fn gap_between(&self, i: usize) -> Option<&GapConstraint> {
        self.gaps.get(i)
    }
}

// ── GrammarLevel ──────────────────────────────────────────────────────────────

/// All patterns induced at one grammar level, plus calibration data.
///
/// `levels[0]` = atom level (base patterns from raw sequences).
/// `levels[1]` = first hierarchical level (patterns over level-0 pattern IDs), etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarLevel {
    /// Patterns induced at this level.
    pub patterns: Vec<Pattern>,
    /// E_norm of each training sequence at this level, used for calibration.
    /// Populated after training converges (Phase 1e).
    pub corpus_e_norms: Vec<f64>,
}

impl GrammarLevel {
    pub fn new(patterns: Vec<Pattern>) -> Self {
        Self {
            patterns,
            corpus_e_norms: Vec::new(),
        }
    }
}

// ── EDistribution ─────────────────────────────────────────────────────────────

/// Calibrated E_norm distribution from training sequences.
///
/// Used to compute `anomaly_percentile` at inference and to gate `is_anomaly`
/// via a configurable threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EDistribution {
    /// E_norm values from training sequences, sorted ascending.
    /// Private — access via `percentile()`.
    sorted_e_norms: Vec<f64>,

    /// Anomaly gate: `E_norm > threshold` → `is_anomaly`.
    /// Default: `0.0` — any uncovered symbol = anomaly (same as original `E > 0` behavior).
    /// Operators can raise this via `Spma::set_anomaly_threshold()`.
    pub threshold: f64,

    /// Per-level sorted E_norm distributions, one vec per grammar level.
    pub level_sorted_e_norms: Vec<Vec<f64>>,

    /// Per-level anomaly thresholds. Falls back to `threshold` when empty or shorter than `level_e_norms`.
    #[serde(default)]
    pub level_thresholds: Vec<f64>,
}

impl EDistribution {
    /// Fraction of training sequences with `E_norm <= e_norm`.
    ///
    /// Returns `0.0` if the distribution is empty (no training data yet).
    pub fn percentile(&self, e_norm: f64) -> f64 {
        if self.sorted_e_norms.is_empty() {
            return 0.0;
        }
        let pos = self.sorted_e_norms.partition_point(|&x| x <= e_norm);
        pos as f64 / self.sorted_e_norms.len() as f64
    }

    /// Fraction of training sequences with `E_norm < e_norm` (strict).
    ///
    /// Use this for anomaly scoring: 0.0 means "at or below all training sequences"
    /// (not anomalous), 1.0 means "worse than every training sequence seen".
    pub fn anomaly_rank(&self, e_norm: f64) -> f64 {
        if self.sorted_e_norms.is_empty() {
            return 0.0;
        }
        let pos = self.sorted_e_norms.partition_point(|&x| x < e_norm);
        pos as f64 / self.sorted_e_norms.len() as f64
    }

    /// Percentile for a specific grammar level.
    pub fn level_percentile(&self, level: usize, e_norm: f64) -> f64 {
        let dist = match self.level_sorted_e_norms.get(level) {
            Some(d) => d,
            None => return 0.0,
        };
        if dist.is_empty() {
            return 0.0;
        }
        let pos = dist.partition_point(|&x| x <= e_norm);
        pos as f64 / dist.len() as f64
    }

    /// E_norm value at the given quantile (e.g. `0.9` for p90).
    /// Returns `0.0` if the distribution is empty.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.sorted_e_norms.is_empty() {
            return 0.0;
        }
        let idx =
            ((q * self.sorted_e_norms.len() as f64) as usize).min(self.sorted_e_norms.len() - 1);
        self.sorted_e_norms[idx]
    }

    /// Populate from a vec of E_norm values (will be sorted in place).
    pub fn fit(mut e_norms: Vec<f64>, threshold: f64, level_e_norms: Vec<Vec<f64>>) -> Self {
        e_norms.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut level_sorted = level_e_norms;
        for v in &mut level_sorted {
            v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        Self {
            sorted_e_norms: e_norms,
            threshold,
            level_sorted_e_norms: level_sorted,
            level_thresholds: Vec::new(),
        }
    }
}

impl Default for EDistribution {
    fn default() -> Self {
        Self {
            sorted_e_norms: Vec::new(),
            threshold: 0.0,
            level_sorted_e_norms: Vec::new(),
            level_thresholds: Vec::new(),
        }
    }
}

// ── Grammar ───────────────────────────────────────────────────────────────────

/// The complete learned grammar.
///
/// `levels[0]` = atom-level patterns.
/// `levels[1..n]` = hierarchical levels.
/// `e_distribution` is populated after training converges (Phase 1e).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grammar {
    pub interner: Interner,
    pub levels: Vec<GrammarLevel>,
    pub e_distribution: EDistribution,
}

impl Grammar {
    pub fn new(interner: Interner) -> Self {
        Self {
            interner,
            levels: Vec::new(),
            e_distribution: EDistribution::default(),
        }
    }

    /// Atom-level patterns (level 0). Returns empty slice if no levels yet.
    pub fn atom_patterns(&self) -> &[Pattern] {
        self.levels
            .first()
            .map(|l| l.patterns.as_slice())
            .unwrap_or(&[])
    }

    /// Patterns at a given level. Returns empty slice if level does not exist.
    pub fn patterns_at(&self, level: usize) -> &[Pattern] {
        self.levels
            .get(level)
            .map(|l| l.patterns.as_slice())
            .unwrap_or(&[])
    }
}

impl Default for Grammar {
    fn default() -> Self {
        Self::new(Interner::new())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_pattern_has_empty_gaps() {
        let p = Pattern::new_contiguous(1, vec![SymbolRef::Atom(0), SymbolRef::Atom(1)], 0);
        assert!(p.is_contiguous());
        assert!(p.gaps.is_empty());
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn gap_pattern_invariant_holds() {
        let symbols = vec![SymbolRef::Atom(0), SymbolRef::Atom(1), SymbolRef::Atom(2)];
        let gaps = vec![GapConstraint::up_to(3), GapConstraint::up_to(2)];
        let p = Pattern::new_with_gaps(2, symbols, gaps, 0);
        assert!(!p.is_contiguous());
        assert_eq!(p.gaps.len(), p.symbols.len() - 1);
    }

    #[test]
    fn gap_between_returns_correct_constraint() {
        let symbols = vec![SymbolRef::Atom(0), SymbolRef::Atom(1), SymbolRef::Atom(2)];
        let gaps = vec![GapConstraint::new(1, 3), GapConstraint::new(0, 2)];
        let p = Pattern::new_with_gaps(3, symbols, gaps, 0);
        assert_eq!(p.gap_between(0), Some(&GapConstraint::new(1, 3)));
        assert_eq!(p.gap_between(1), Some(&GapConstraint::new(0, 2)));
        assert_eq!(p.gap_between(2), None);
    }

    #[test]
    fn e_distribution_percentile_empty() {
        let dist = EDistribution::default();
        assert_eq!(dist.percentile(1.0), 0.0);
        assert_eq!(dist.threshold, 0.0);
    }

    #[test]
    fn e_distribution_percentile_correct() {
        let dist = EDistribution::fit(vec![0.1, 0.3, 0.5, 0.7, 0.9], 0.0, vec![]);
        // 3 values <= 0.5 out of 5 → 0.6
        assert!((dist.percentile(0.5) - 0.6).abs() < 1e-10);
        // all values <= 1.0 → 1.0
        assert!((dist.percentile(1.0) - 1.0).abs() < 1e-10);
        // no values <= 0.0 → 0.0
        assert!((dist.percentile(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn e_distribution_fit_sorts_input() {
        let dist = EDistribution::fit(vec![0.9, 0.1, 0.5], 0.0, vec![]);
        // After fit, sorted_e_norms must be ascending — verify via percentile monotonicity
        assert!(dist.percentile(0.1) <= dist.percentile(0.5));
        assert!(dist.percentile(0.5) <= dist.percentile(0.9));
    }

    #[test]
    fn grammar_new_has_no_levels() {
        let g = Grammar::new(Interner::new());
        assert!(g.levels.is_empty());
        assert!(g.atom_patterns().is_empty());
    }

    #[test]
    fn grammar_patterns_at_missing_level_returns_empty() {
        let g = Grammar::new(Interner::new());
        assert!(g.patterns_at(99).is_empty());
    }

    #[test]
    fn interner_serde_roundtrip() {
        let mut interner = Interner::new();
        interner.intern("hello");
        interner.intern("world");
        let json = serde_json::to_string(&interner).unwrap();
        let restored: Interner = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), 2);
        assert_eq!(restored.name(0), "hello");
        assert_eq!(restored.name(1), "world");
    }

    #[test]
    fn grammar_serde_roundtrip() {
        let mut interner = Interner::new();
        let a = interner.intern("A");
        let b = interner.intern("B");
        let pattern = Pattern::new_contiguous(1, vec![SymbolRef::Atom(a), SymbolRef::Atom(b)], 0);
        let mut g = Grammar::new(interner);
        g.levels.push(GrammarLevel::new(vec![pattern]));

        let json = serde_json::to_string(&g).unwrap();
        let restored: Grammar = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.interner.len(), 2);
        assert_eq!(restored.levels.len(), 1);
        assert_eq!(restored.atom_patterns().len(), 1);
        assert_eq!(restored.atom_patterns()[0].id, 1);
    }
}
