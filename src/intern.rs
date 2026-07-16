use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interner {
    map: HashMap<String, u32>,
    names: Vec<String>,
}

impl Interner {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            names: Vec::new(),
        }
    }

    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        assert!(
            self.names.len() < u32::MAX as usize,
            "Interner overflow: too many unique symbols"
        );
        let id = self.names.len() as u32;
        self.names.push(s.to_owned());
        self.map.insert(s.to_owned(), id);
        id
    }

    pub fn name(&self, id: u32) -> &str {
        self.names
            .get(id as usize)
            .expect("intern ID out of bounds — ID from different Interner instance?")
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Look up an existing symbol ID without inserting. Returns `None` if unknown.
    pub fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}
