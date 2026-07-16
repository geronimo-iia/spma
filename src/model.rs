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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolRef {
    Atom(u32),
    Pattern(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: SymbolRef,
    pub symbol_type: SymbolType,
    pub status: SymbolStatus,
    pub frequency: u32,
    pub bit_cost: f64,
    pub position: i32,
}

impl Symbol {
    pub fn new(name: u32) -> Self {
        Self {
            name: SymbolRef::Atom(name),
            symbol_type: SymbolType::DataSymbol,
            status: SymbolStatus::Contents,
            frequency: 1,
            bit_cost: 0.0,
            position: -1,
        }
    }

    pub fn new_pattern_ref(pattern_id: u32) -> Self {
        Self {
            name: SymbolRef::Pattern(pattern_id),
            symbol_type: SymbolType::DataSymbol,
            status: SymbolStatus::Contents,
            frequency: 1,
            bit_cost: 0.0,
            position: -1,
        }
    }

    pub fn atom_id(&self) -> Option<u32> {
        match self.name {
            SymbolRef::Atom(id) => Some(id),
            _ => None,
        }
    }

    pub fn pattern_id(&self) -> Option<u32> {
        match self.name {
            SymbolRef::Pattern(id) => Some(id),
            _ => None,
        }
    }

    /// Returns the inner u32 regardless of variant. Used for cost-table indexing.
    pub fn raw_id(&self) -> u32 {
        match self.name {
            SymbolRef::Atom(id) | SymbolRef::Pattern(id) => id,
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
    match sym.name {
        SymbolRef::Atom(id) => interner.name(id).to_owned(),
        SymbolRef::Pattern(pid) => format!("[pat:{}]", pid),
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
            .map(|s| match s.name {
                SymbolRef::Atom(id) => interner.name(id).to_owned(),
                SymbolRef::Pattern(pid) => format!("[pat:{}]", pid),
            })
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
