use spma::*;

fn make_interner_and_symbol(name: &str) -> (Interner, Symbol) {
    let mut interner = Interner::new();
    let id = interner.intern(name);
    (interner, Symbol::new(id))
}

#[test]
fn test_symbol_creation() {
    let (interner, symbol) = make_interner_and_symbol("test");
    assert_eq!(interner.name(symbol.raw_id()), "test");
    assert_eq!(symbol.symbol_type, SymbolType::DataSymbol);
    assert_eq!(symbol.status, SymbolStatus::Contents);
    assert_eq!(symbol.frequency, 1);
}

#[test]
fn test_symbol_matching() {
    let mut interner = Interner::new();
    let cat_id = interner.intern("cat");
    let dog_id = interner.intern("dog");

    let sym1 = Symbol::new(cat_id);
    let sym2 = Symbol::new(cat_id);
    let sym3 = Symbol::new(dog_id);

    assert!(sym1.matches(&sym2));
    assert!(!sym1.matches(&sym3));
}

#[test]
fn test_pattern_creation() {
    let mut interner = Interner::new();
    let the_id = interner.intern("the");
    let cat_id = interner.intern("cat");

    let symbols = vec![Symbol::new(the_id), Symbol::new(cat_id)];
    let pattern = Pattern::new(symbols, 1);

    assert_eq!(pattern.pattern_id, 1);
    assert_eq!(pattern.len(), 2);
    assert!(!pattern.is_empty());
    assert_eq!(pattern.get_symbol_names(&interner), vec!["the", "cat"]);
}

#[test]
fn test_pattern_cost_calculation() {
    let mut interner = Interner::new();
    let id1 = interner.intern("test1");
    let id2 = interner.intern("test2");

    let mut symbols = vec![Symbol::new(id1), Symbol::new(id2)];
    symbols[0].bit_cost = 2.0;
    symbols[1].bit_cost = 3.0;

    let pattern = Pattern::new(symbols, 1);
    assert_eq!(pattern.compute_total_cost(), 5.0);
}

// ── Interner ──────────────────────────────────────────────────────────────

#[test]
fn interner_same_string_same_id() {
    let mut i = Interner::new();
    let a = i.intern("foo");
    let b = i.intern("foo");
    assert_eq!(a, b);
    assert_eq!(i.len(), 1);
}

#[test]
fn interner_different_strings_different_ids() {
    let mut i = Interner::new();
    let a = i.intern("foo");
    let b = i.intern("bar");
    assert_ne!(a, b);
    assert_eq!(i.len(), 2);
}

#[test]
fn interner_name_roundtrip() {
    let mut i = Interner::new();
    let id = i.intern("hello");
    assert_eq!(i.name(id), "hello");
}

#[test]
#[should_panic(expected = "intern ID out of bounds")]
fn interner_name_out_of_bounds_panics() {
    let i = Interner::new();
    i.name(99);
}

// ── assign_symbol_types ───────────────────────────────────────────────────

#[test]
fn assign_symbol_types_promotes_identification_symbols() {
    let mut sp = SpmaEngine::new();
    let a = sp.interner.intern("A");
    let b = sp.interner.intern("B");

    let mut sym_a = Symbol::new(a);
    sym_a.status = SymbolStatus::Contents;
    let mut sym_b = Symbol::new(b);
    sym_b.status = SymbolStatus::Identification;

    sp.new_patterns = vec![Pattern::new(vec![sym_a, sym_b], 1)];
    sp.assign_symbol_types();

    let b_in_pat = sp.new_patterns[0].symbols.iter().find(|s| s.raw_id() == b).unwrap();
    assert_eq!(
        b_in_pat.symbol_type,
        SymbolType::ContextSymbol,
        "Identification symbol should be promoted to ContextSymbol"
    );

    let a_in_pat = sp.new_patterns[0].symbols.iter().find(|s| s.raw_id() == a).unwrap();
    assert_eq!(
        a_in_pat.symbol_type,
        SymbolType::DataSymbol,
        "Contents symbol should stay DataSymbol"
    );
}

#[test]
fn assign_symbol_types_no_identification_no_change() {
    let mut sp = SpmaEngine::new();
    let a = sp.interner.intern("A");
    sp.new_patterns = vec![Pattern::new(vec![Symbol::new(a)], 1)];
    sp.assign_symbol_types();
    assert_eq!(sp.new_patterns[0].symbols[0].symbol_type, SymbolType::DataSymbol);
}

// ── compute_t_ge ─────────────────────────────────────────────────────────

#[test]
fn compute_t_ge_overlapping_symbol_ids_in_new_and_old() {
    let costs = vec![2.0, 3.0];
    let new = &[0u32, 1];
    let old: &[&[u32]] = &[&[0u32]];
    let covered = &[true, false];
    let (g, e, t) = compute_t_ge(new, old, &costs, covered);
    assert!((g - 2.0).abs() < 1e-9, "G={g}");
    assert!((e - 3.0).abs() < 1e-9, "E={e}");
    assert!((t - 5.0).abs() < 1e-9, "T={t}");
}

#[test]
fn compute_t_ge_empty_old_all_e() {
    let costs = vec![1.0, 2.0, 3.0];
    let new = &[0u32, 1, 2];
    let old: &[&[u32]] = &[];
    let covered = &[false, false, false];
    let (g, e, t) = compute_t_ge(new, old, &costs, covered);
    assert!((g - 0.0).abs() < 1e-9);
    assert!((e - 6.0).abs() < 1e-9);
    assert!((t - 6.0).abs() < 1e-9);
}

#[test]
fn compute_t_ge_all_covered_no_e() {
    let costs = vec![1.0, 2.0];
    let new = &[0u32, 1];
    let old: &[&[u32]] = &[&[0u32, 1]];
    let covered = &[true, true];
    let (g, e, t) = compute_t_ge(new, old, &costs, covered);
    assert!((g - 3.0).abs() < 1e-9);
    assert!((e - 0.0).abs() < 1e-9);
    assert!((t - 3.0).abs() < 1e-9);
}

#[test]
fn test_v1_shannon_bit_costs() {
    // Patterns: ["a b c", "a b d", "a b e"]
    // Freqs: a=3, b=3, c=1, d=1, e=1, total=9
    // Expected: a,b cost = -log2(3/9) ≈ 1.585; c,d,e cost = -log2(1/9) ≈ 3.170
    let mut sp = SpmaEngine::new();

    let (a, b, c, d, e) = {
        let interner = &mut sp.interner;
        let a = interner.intern("a");
        let b = interner.intern("b");
        let c = interner.intern("c");
        let d = interner.intern("d");
        let e = interner.intern("e");
        (a, b, c, d, e)
    };

    fn make_pat(ids: &[u32], pid: u32) -> Pattern {
        let symbols = ids.iter().map(|&id| Symbol::new(id)).collect();
        Pattern::new(symbols, pid)
    }

    let mut patterns = vec![
        make_pat(&[a, b, c], 1),
        make_pat(&[a, b, d], 2),
        make_pat(&[a, b, e], 3),
    ];

    sp.calculate_symbol_frequencies(&patterns);
    sp.assign_symbol_costs(&mut patterns);

    let cost_of = |id: u32| -> f64 {
        patterns
            .iter()
            .flat_map(|p| p.symbols.iter())
            .find(|s| s.raw_id() == id)
            .expect("symbol not found in patterns")
            .bit_cost
    };

    let expected_ab = -(3.0_f64 / 9.0).log2();
    let expected_cde = -(1.0_f64 / 9.0).log2();

    assert!((cost_of(a) - expected_ab).abs() < 0.001, "cost(a)={}", cost_of(a));
    assert!((cost_of(b) - expected_ab).abs() < 0.001, "cost(b)={}", cost_of(b));
    assert!((cost_of(c) - expected_cde).abs() < 0.001, "cost(c)={}", cost_of(c));
    assert!((cost_of(d) - expected_cde).abs() < 0.001, "cost(d)={}", cost_of(d));
    assert!((cost_of(e) - expected_cde).abs() < 0.001, "cost(e)={}", cost_of(e));
}

#[test]
fn test_v2_t_ge_formula() {
    use spma::compute_t_ge;

    let costs = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    // Case A: full match
    let new_a = &[0u32, 1, 2];
    let old_a: &[&[u32]] = &[&[0u32, 1, 2]];
    let covered_a = &[true, true, true];
    let (g, e, t) = compute_t_ge(new_a, old_a, &costs, covered_a);
    assert!((g - 6.0).abs() < 1e-9, "Case A: G={g}");
    assert!((e - 0.0).abs() < 1e-9, "Case A: E={e}");
    assert!((t - 6.0).abs() < 1e-9, "Case A: T={t}");

    // Case B: no match
    let new_b = &[3u32, 4];
    let old_b: &[&[u32]] = &[];
    let covered_b = &[false, false];
    let (g, e, t) = compute_t_ge(new_b, old_b, &costs, covered_b);
    assert!((g - 0.0).abs() < 1e-9, "Case B: G={g}");
    assert!((e - 9.0).abs() < 1e-9, "Case B: E={e}");
    assert!((t - 9.0).abs() < 1e-9, "Case B: T={t}");

    // Case C: partial match
    let new_c = &[0u32, 1, 3, 2];
    let old_c: &[&[u32]] = &[&[0u32, 1, 2]];
    let covered_c = &[true, true, false, true];
    let (g, e, t) = compute_t_ge(new_c, old_c, &costs, covered_c);
    assert!((g - 6.0).abs() < 1e-9, "Case C: G={g}");
    assert!((e - 4.0).abs() < 1e-9, "Case C: E={e}");
    assert!((t - 10.0).abs() < 1e-9, "Case C: T={t}");
}
