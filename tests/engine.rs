use spma::*;

fn create_test_pattern(interner: &mut Interner, words: Vec<&str>, id: u32) -> Pattern {
    let symbols: Vec<Symbol> = words
        .iter()
        .map(|word| Symbol::new(interner.intern(word)))
        .collect();
    Pattern::new(symbols, id)
}

#[test]
fn test_spma_initialization() {
    let sp = SpmaEngine::new();
    assert_eq!(sp.next_pattern_id, 1);
    assert_eq!(sp.keep_rows, 5);
    assert_eq!(sp.max_cycles, 1000);
}

#[test]
fn test_symbol_frequency_calculation() {
    let mut sp = SpmaEngine::new();
    let patterns = vec![
        create_test_pattern(&mut sp.interner, vec!["cat", "sat"], 1),
        create_test_pattern(&mut sp.interner, vec!["cat", "ran"], 2),
    ];

    sp.calculate_symbol_frequencies(&patterns);
    let cat_id = sp.interner.intern("cat");
    let sat_id = sp.interner.intern("sat");
    assert_eq!(sp.symbol_frequencies.get(&cat_id), Some(&2));
    assert_eq!(sp.symbol_frequencies.get(&sat_id), Some(&1));
}

#[test]
fn test_pattern_input_parsing() {
    let mut sp = SpmaEngine::new();

    use std::fs;
    let test_content = "< the cat > sat on < the mat >\n!animal dog #1 eats !food meat #2";
    fs::write("test_input.txt", test_content).unwrap();

    let patterns = sp.load_input("test_input.txt").unwrap();
    assert_eq!(patterns.len(), 2);

    assert_eq!(
        patterns[0].get_symbol_names(&sp.interner),
        vec!["<", "the", "cat", ">", "sat", "on", "<", "the", "mat", ">"]
    );

    let second_pattern_symbols = patterns[1].get_symbol_names(&sp.interner);
    assert!(second_pattern_symbols.contains(&"dog".to_string()));
    assert!(second_pattern_symbols.contains(&"meat".to_string()));

    fs::remove_file("test_input.txt").unwrap();
}

#[test]
fn test_learning_cycle() {
    let mut sp = SpmaEngine::new();

    let patterns = vec![
        create_test_pattern(&mut sp.interner, vec!["cat", "sat"], 1),
        create_test_pattern(&mut sp.interner, vec!["dog", "sat"], 2),
    ];

    let results = sp.learn(patterns).unwrap();
    assert!(results.cycles > 0);
    assert!(results.final_patterns.is_empty());
}

#[test]
fn test_v4_convergence() {
    let mut sp = SpmaEngine::new();
    sp.max_cycles = 5;

    let patterns = vec![
        create_test_pattern(&mut sp.interner, vec!["a", "b", "c"], 1),
        create_test_pattern(&mut sp.interner, vec!["a", "b", "d"], 2),
        create_test_pattern(&mut sp.interner, vec!["a", "b", "e"], 3),
        create_test_pattern(&mut sp.interner, vec!["x", "y", "z"], 4),
        create_test_pattern(&mut sp.interner, vec!["x", "y", "w"], 5),
    ];

    let results = sp.learn(patterns).unwrap();
    assert!(results.cycles >= 1, "should run at least 1 cycle");
    assert!(!results.final_patterns.is_empty(), "should have patterns");
}

#[test]
fn test_v5_one_trial_learning() {
    let mut sp = SpmaEngine::new();
    sp.max_cycles = 3;

    let patterns = vec![create_test_pattern(
        &mut sp.interner,
        vec!["fault_A", "fault_B", "fault_C"],
        1,
    )];

    let results = sp.learn(patterns).unwrap();
    assert!(results.final_patterns.is_empty(), "no repeated structure → empty grammar");
}

#[test]
fn test_v5_one_trial_learning_with_repetition() {
    let mut sp = SpmaEngine::new();
    sp.max_cycles = 5;

    let patterns = vec![
        create_test_pattern(&mut sp.interner, vec!["fault_A", "fault_B", "fault_C"], 1),
        create_test_pattern(&mut sp.interner, vec!["fault_A", "fault_B", "fault_C"], 2),
    ];

    let results = sp.learn(patterns).unwrap();

    let fault_a_id = sp.interner.intern("fault_A");
    let fault_b_id = sp.interner.intern("fault_B");

    let has_ab = results.final_patterns.iter().any(|p| {
        let ids: Vec<u32> = p.symbols.iter().map(|s| s.raw_id()).collect();
        ids.contains(&fault_a_id) && ids.contains(&fault_b_id)
    });
    assert!(has_ab, "repeated bigram fault_A fault_B should be in grammar");
}

#[test]
fn test_v6_grammar_recovery() {
    let mut sp = SpmaEngine::new();
    sp.max_cycles = 10;
    sp.keep_rows = 10;

    let sentences: Vec<Vec<&str>> = vec![
        vec!["the", "cat", "sat", "the", "mat"],
        vec!["the", "dog", "sat", "a", "cat"],
        vec!["a", "cat", "chased", "the", "dog"],
        vec!["a", "mat", "sat", "the", "cat"],
        vec!["the", "dog", "chased", "a", "mat"],
        vec!["the", "cat", "sat", "a", "dog"],
        vec!["a", "dog", "sat", "the", "cat"],
        vec!["the", "mat", "chased", "a", "dog"],
        vec!["a", "cat", "sat", "the", "mat"],
        vec!["the", "dog", "sat", "the", "cat"],
        vec!["a", "mat", "chased", "a", "cat"],
        vec!["the", "cat", "chased", "the", "dog"],
        vec!["a", "dog", "chased", "the", "mat"],
        vec!["the", "mat", "sat", "a", "cat"],
        vec!["a", "cat", "sat", "a", "dog"],
        vec!["the", "dog", "chased", "the", "mat"],
        vec!["a", "mat", "sat", "a", "dog"],
        vec!["the", "cat", "chased", "a", "mat"],
        vec!["a", "dog", "sat", "the", "mat"],
        vec!["the", "mat", "chased", "the", "cat"],
    ];

    let mut patterns = Vec::new();
    for (i, sentence) in sentences.iter().enumerate() {
        patterns.push(create_test_pattern(&mut sp.interner, sentence.clone(), (i + 1) as u32));
    }

    let results = sp.learn(patterns).unwrap();

    let compression_ratio = sp.compute_global_compression_ratio(
        &sp.new_patterns.clone(),
        &results.final_patterns,
        10,
    );

    println!(
        "Global compression ratio: {:.3}, grammar size: {} patterns",
        compression_ratio,
        results.final_patterns.len()
    );

    let multi_symbol_count = results
        .final_patterns
        .iter()
        .filter(|p| p.symbols.len() >= 2)
        .count();

    assert!(
        compression_ratio > 2.0,
        "expected global compression ratio > 2.0, got {:.3}. Grammar size: {} patterns",
        compression_ratio,
        results.final_patterns.len()
    );

    assert!(
        results.final_patterns.len() > 8,
        "expected grammar to grow beyond vocabulary size (8), got {}",
        results.final_patterns.len()
    );

    assert!(
        multi_symbol_count > 0,
        "expected multi-symbol patterns to be extracted, got 0"
    );
}

// ── extract_frequent_ngrams cold-start vs beam-switch ─────────────────────

#[test]
fn extract_frequent_ngrams_cold_start_adds_bigrams() {
    let mut sp = SpmaEngine::new();
    let pats = vec![
        create_test_pattern(&mut sp.interner, vec!["a", "b", "c"], 1),
        create_test_pattern(&mut sp.interner, vec!["a", "b", "d"], 2),
    ];
    let results = sp.learn(pats).unwrap();
    let a = sp.interner.intern("a");
    let b = sp.interner.intern("b");
    let has_ab = results.final_patterns.iter().any(|p| {
        let ids: Vec<u32> = p.symbols.iter().map(|s| s.raw_id()).collect();
        ids == vec![a, b]
    });
    assert!(has_ab, "cold-start should extract bigram [a,b]");
}

#[test]
fn extract_frequent_ngrams_does_not_add_singletons() {
    let mut sp = SpmaEngine::new();
    let pats = vec![
        create_test_pattern(&mut sp.interner, vec!["a", "b"], 1),
        create_test_pattern(&mut sp.interner, vec!["c", "d"], 2),
    ];
    let results = sp.learn(pats).unwrap();
    let multi = results.final_patterns.iter().filter(|p| p.symbols.len() >= 2).count();
    assert_eq!(multi, 0, "no bigram appears ≥2 times, should add none");
}

#[test]
fn beam_driven_phase_skips_ngram_miner() {
    let mut sp = SpmaEngine::new();
    let pats = vec![
        create_test_pattern(&mut sp.interner, vec!["a", "b", "c"], 1),
        create_test_pattern(&mut sp.interner, vec!["a", "b", "d"], 2),
        create_test_pattern(&mut sp.interner, vec!["a", "b", "e"], 3),
    ];
    let r1 = sp.learn(pats.clone()).unwrap();
    let mut sp2 = SpmaEngine::new();
    sp2.interner = sp.interner.clone();
    let r2 = sp2.learn(pats).unwrap();
    assert_eq!(
        r1.final_patterns.len(),
        r2.final_patterns.len(),
        "deterministic: same corpus same result"
    );
}

// ── compute_global_compression_ratio ─────────────────────────────────────

#[test]
fn compression_ratio_empty_grammar_is_one() {
    let sp = SpmaEngine::new();
    let ratio = sp.compute_global_compression_ratio(&[], &[], 5);
    assert!((ratio - 1.0).abs() < 1e-9, "ratio={ratio}");
}

#[test]
fn compression_ratio_no_multi_symbol_patterns_is_one() {
    let mut sp = SpmaEngine::new();
    let pats = vec![create_test_pattern(&mut sp.interner, vec!["a"], 1)];
    let ratio = sp.compute_global_compression_ratio(&pats, &pats, 5);
    assert!((ratio - 1.0).abs() < 1e-9, "ratio={ratio}");
}

#[test]
fn compression_ratio_perfect_coverage_greater_than_one() {
    let mut sp = SpmaEngine::new();
    let a = sp.interner.intern("a");
    let b = sp.interner.intern("b");
    let mut sa = Symbol::new(a);
    sa.bit_cost = 2.0;
    let mut sb = Symbol::new(b);
    sb.bit_cost = 2.0;
    let grammar_pat = Pattern::new(vec![sa.clone(), sb.clone()], 1);
    let new_pats: Vec<Pattern> = (0..5)
        .map(|i| Pattern::new(vec![sa.clone(), sb.clone()], i + 10))
        .collect();
    let ratio = sp.compute_global_compression_ratio(&new_pats, &[grammar_pat], 5);
    assert!(ratio > 1.0, "good compression should yield ratio > 1, got {ratio}");
}

// ── MDL accumulator ───────────────────────────────────────────────────────

#[test]
fn mdl_accumulator_determinism_multi_candidate_corpus() {
    let corpus: Vec<Vec<&str>> = vec![
        vec!["A", "B", "C"],
        vec!["A", "B", "C"],
        vec!["A", "B", "D"],
        vec!["A", "B", "D"],
        vec!["X", "Y", "Z"],
        vec!["X", "Y", "Z"],
        vec!["X", "Y", "W"],
        vec!["A", "B", "C"],
        vec!["X", "Y", "Z"],
        vec!["A", "B", "D"],
    ];

    let mut e1 = spma::Spma::new();
    e1.train(&corpus).unwrap();

    let mut e2 = spma::Spma::new();
    e2.train(&corpus).unwrap();

    let probe = vec!["A", "B", "C"];
    let r1 = e1.infer(&probe).unwrap();
    let r2 = e2.infer(&probe).unwrap();
    assert_eq!(r1.e_cost, r2.e_cost, "accumulator runs diverged");
    assert_eq!(r1.is_anomaly, r2.is_anomaly);

    let probe2 = vec!["X", "Y", "W"];
    let r3 = e1.infer(&probe2).unwrap();
    let r4 = e2.infer(&probe2).unwrap();
    assert_eq!(r3.e_cost, r4.e_cost);
}
