use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::Interner;

/// Symbol types recognized by the SPMA system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolType {
    DataSymbol,
    ContextSymbol,
    LeftBracket,
    RightBracket,
    UniqueIdSymbol,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolStatus {
    Identification,
    Contents,
    BoundaryMarker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignmentType {
    FullA,   // Both patterns fully matched
    FullB,   // Old patterns fully matched, New partially
    FullC,   // Old patterns fully matched, New with gaps
    Partial, // Partial matching
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: u32,
    pub symbol_type: SymbolType,
    pub status: SymbolStatus,
    pub frequency: u32,
    pub bit_cost: f64,
    pub position: i32,
}

impl Symbol {
    pub fn new(name: u32) -> Self {
        Self {
            name,
            symbol_type: SymbolType::DataSymbol,
            status: SymbolStatus::Contents,
            frequency: 1,
            bit_cost: 0.0,
            position: -1,
        }
    }

    pub fn matches(&self, other: &Symbol) -> bool {
        self.name == other.name
    }
}

impl Hash for Symbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Symbol {}

pub fn format_symbol(sym: &Symbol, interner: &Interner) -> String {
    interner.name(sym.name).to_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentElement {
    pub symbol: Option<Symbol>,
    pub original_pattern_id: Option<u32>,
    pub same_column_above: i32,
    pub same_column_below: i32,
    pub original_position: i32,
}

impl Default for AlignmentElement {
    fn default() -> Self {
        Self {
            symbol: None,
            original_pattern_id: None,
            same_column_above: -1,
            same_column_below: -1,
            original_position: -1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub symbols: Vec<Symbol>,
    pub pattern_id: u32,
    pub frequency: u32,
    pub compression_difference: f64,
    pub encoding_cost: f64,
    pub new_symbols_cost: f64,
    pub compression_ratio: f64,
    pub origin: String,
    pub keep: bool,
    pub new_this_cycle: bool,
}

impl Pattern {
    pub fn new(symbols: Vec<Symbol>, pattern_id: u32) -> Self {
        Self {
            symbols,
            pattern_id,
            frequency: 1,
            compression_difference: 0.0,
            encoding_cost: 0.0,
            new_symbols_cost: 0.0,
            compression_ratio: 0.0,
            origin: "BASIC_PATTERN".to_string(),
            keep: false,
            new_this_cycle: true,
        }
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    pub fn get_symbol_names(&self, interner: &Interner) -> Vec<String> {
        self.symbols
            .iter()
            .map(|s| interner.name(s.name).to_owned())
            .collect()
    }

    pub fn compute_total_cost(&self) -> f64 {
        self.symbols.iter().map(|s| s.bit_cost).sum()
    }

    pub fn has_brackets(&self) -> bool {
        self.symbols.len() >= 2
            && self.symbols[0].symbol_type == SymbolType::LeftBracket
            && self.symbols[self.symbols.len() - 1].symbol_type == SymbolType::RightBracket
    }

    pub fn is_abstract_pattern(&self) -> bool {
        self.symbols
            .iter()
            .all(|s| s.symbol_type != SymbolType::DataSymbol)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alignment {
    pub patterns: Vec<Pattern>,
    pub alignment_id: u32,
    pub columns: Vec<Vec<AlignmentElement>>,
    pub compression_ratio: f64,
    pub compression_difference: f64,
    pub encoding_cost: f64,
    pub new_symbols_cost: f64,
    pub degree_of_matching: AlignmentType,
    pub leaf_node_id: i32,
}

impl Alignment {
    pub fn new(patterns: Vec<Pattern>, alignment_id: u32) -> Self {
        Self {
            patterns,
            alignment_id,
            columns: Vec::new(),
            compression_ratio: 0.0,
            compression_difference: 0.0,
            encoding_cost: 0.0,
            new_symbols_cost: 0.0,
            degree_of_matching: AlignmentType::Partial,
            leaf_node_id: -1,
        }
    }

    pub fn find_degree_of_matching(&mut self) {
        if self.patterns.is_empty() {
            return;
        }

        let mut new_fully_matched = true;
        let mut old_fully_matched = true;

        for col in &self.columns {
            if !col.is_empty() && col[0].symbol.is_some() {
                let has_match = col[1..].iter().any(|elem| {
                    elem.symbol
                        .as_ref()
                        .is_some_and(|s| col[0].symbol.as_ref().unwrap().matches(s))
                });
                if !has_match {
                    new_fully_matched = false;
                }
            }
        }

        for pattern_idx in 1..self.patterns.len() {
            let pattern = &self.patterns[pattern_idx];
            let mut pattern_symbols_matched = 0;

            for col in &self.columns {
                if pattern_idx < col.len()
                    && col[pattern_idx].symbol.is_some()
                    && col[pattern_idx].same_column_above >= 0
                {
                    pattern_symbols_matched += 1;
                }
            }

            if pattern_symbols_matched < pattern.symbols.len() {
                old_fully_matched = false;
            }
        }

        self.degree_of_matching = match (new_fully_matched, old_fully_matched) {
            (true, true) => AlignmentType::FullA,
            (false, true) => AlignmentType::FullB,
            (_, false) => AlignmentType::Partial,
        };
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitNode {
    pub node_id: u32,
    pub driving_pattern_id: u32,
    pub target_pattern_id: u32,
    pub driving_symbol: Symbol,
    pub target_symbol: Symbol,
    pub driving_position: usize,
    pub target_position: usize,
    pub compression_difference: f64,
    pub compression_ratio: f64,
    pub children: Vec<u32>,
    pub parent: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grammar {
    pub grammar_id: u32,
    pub patterns: Vec<Pattern>,
    pub grammar_size: f64,
    pub encoding_size: f64,
    pub total_score: f64,
}

impl Grammar {
    pub fn new(grammar_id: u32, patterns: Vec<Pattern>) -> Self {
        let mut grammar = Self {
            grammar_id,
            patterns,
            grammar_size: 0.0,
            encoding_size: 0.0,
            total_score: 0.0,
        };
        grammar.compute_grammar_size();
        grammar
    }

    pub fn compute_grammar_size(&mut self) {
        self.grammar_size = self.patterns.iter().map(|p| p.compute_total_cost()).sum();
    }

    pub fn compute_encoding_size(&mut self, corpus_patterns: &[Pattern]) {
        self.encoding_size = corpus_patterns.iter().map(|p| p.compute_total_cost()).sum();
    }

    pub fn compute_total_score(&mut self) {
        self.total_score = self.grammar_size + self.encoding_size;
    }
}

/// Compute the T=G+E decomposition for a multiple alignment.
pub fn compute_t_ge(
    new_pattern: &[u32],
    old_patterns: &[&[u32]],
    costs: &[f64],
    covered_positions: &[bool],
) -> (f64, f64, f64) {
    debug_assert!(
        old_patterns
            .iter()
            .flat_map(|p| p.iter())
            .all(|&id| (id as usize) < costs.len()),
        "old_patterns contain symbol id >= costs.len()"
    );
    debug_assert!(
        new_pattern.iter().all(|&id| (id as usize) < costs.len()),
        "new_pattern contains symbol id >= costs.len()"
    );
    let g: f64 = old_patterns
        .iter()
        .flat_map(|p| p.iter())
        .map(|&id| costs[id as usize])
        .sum();
    let e: f64 = new_pattern
        .iter()
        .enumerate()
        .filter(|&(i, _)| !covered_positions[i])
        .map(|(_, &id)| costs[id as usize])
        .sum();
    let t = g + e;
    (g, e, t)
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_symbol_types() {
        let mut interner = Interner::new();
        let id = interner.intern("test");
        let mut symbol = Symbol::new(id);
        assert_eq!(symbol.symbol_type, SymbolType::DataSymbol);

        symbol.symbol_type = SymbolType::ContextSymbol;
        assert_eq!(symbol.symbol_type, SymbolType::ContextSymbol);
    }

    #[test]
    fn test_pattern_brackets() {
        let mut interner = Interner::new();
        let symbols = vec![
            create_bracket_symbol(&mut interner, "<"),
            Symbol::new(interner.intern("content")),
            create_bracket_symbol(&mut interner, ">"),
        ];
        let pattern = Pattern::new(symbols, 1);
        assert!(pattern.has_brackets());
    }

    #[test]
    fn test_alignment_element_default() {
        let element = AlignmentElement::default();
        assert!(element.symbol.is_none());
        assert_eq!(element.same_column_above, -1);
        assert_eq!(element.same_column_below, -1);
    }

    #[test]
    fn test_alignment_type_enum() {
        let alignment_type = AlignmentType::FullA;
        assert_eq!(alignment_type, AlignmentType::FullA);
        assert_ne!(alignment_type, AlignmentType::Partial);
    }

    fn create_bracket_symbol(interner: &mut Interner, name: &str) -> Symbol {
        let id = interner.intern(name);
        let mut symbol = Symbol::new(id);
        symbol.symbol_type = match name {
            "<" => SymbolType::LeftBracket,
            ">" => SymbolType::RightBracket,
            _ => SymbolType::DataSymbol,
        };
        symbol.status = SymbolStatus::BoundaryMarker;
        symbol
    }
}
