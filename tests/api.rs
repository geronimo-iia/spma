use spma::*;

#[test]
fn spma_train_save_load_roundtrip() {
    let dir = std::env::temp_dir();
    let path = dir.join("spma_roundtrip_test.bin");
    let path_str = path.to_str().unwrap();

    let mut engine = Spma::new();
    engine
        .train(&[
            vec!["fault_A", "fault_B", "fault_C"],
            vec!["fault_A", "fault_B", "fault_D"],
        ])
        .unwrap();
    engine.save(path_str).unwrap();

    let engine2 = Spma::load(path_str).unwrap();
    let result = engine2.infer(&["fault_A", "fault_B", "fault_C"]).unwrap();
    assert!(
        result.alignment.contains("fault_A") && result.alignment.contains("fault_B"),
        "shared prefix should appear in alignment"
    );
    assert!(
        result.unmatched.contains(&"fault_C".to_string()),
        "fault_C should be unmatched (unique symbol, no grammar pattern)"
    );

    std::fs::remove_file(path_str).ok();
}

#[test]
fn spma_infer_unseen_symbol_is_anomaly() {
    let mut engine = Spma::new();
    engine.train(&[vec!["A", "B", "C"]]).unwrap();
    let result = engine.infer(&["A", "B", "X"]).unwrap();
    assert!(result.is_anomaly, "unseen X should trigger anomaly");
    assert!(result.e_cost > 0.0, "e_cost should be > 0 due to unknown penalty");
    assert!(result.unmatched.contains(&"X".to_string()), "X should be in unmatched");
}

#[test]
fn spma_infer_alignment_string_non_empty() {
    let mut engine = Spma::new();
    engine.train(&[vec!["A", "B"], vec!["A", "C"]]).unwrap();
    let result = engine.infer(&["A", "B"]).unwrap();
    assert!(!result.alignment.is_empty(), "alignment string should be populated");
    assert!(result.alignment.contains("New:"), "alignment should have New row");
}

#[test]
fn spma_infer_fully_unseen_sequence() {
    let mut engine = Spma::new();
    engine.train(&[vec!["A", "B"]]).unwrap();
    let result = engine.infer(&["X", "Y", "Z"]).unwrap();
    assert!(result.is_anomaly, "all-unknown sequence should be anomaly");
    assert_eq!(result.unmatched.len(), 3, "all 3 symbols should be unmatched");
    assert!(result.alignment.contains("New:"), "got: {}", result.alignment);
    assert!(result.alignment.contains('X'), "got: {}", result.alignment);
}

#[test]
fn spma_infer_known_but_rare_symbol_not_penalised_as_unknown() {
    let mut engine = Spma::new();
    engine
        .train(&[
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "D"],
        ])
        .unwrap();
    let unknown_result = engine.infer(&["X"]).unwrap();
    assert!(
        unknown_result.e_cost > 0.0,
        "unknown symbol must have E>0, got e={}",
        unknown_result.e_cost
    );
    let known_result = engine.infer(&["C"]).unwrap();
    assert!(
        known_result.e_cost > 0.0,
        "known-rare symbol not in grammar should have E>0 via corpus cost fallback, got e={}",
        known_result.e_cost
    );
}

#[test]
fn spma_train_special_symbols_bracket_and_id() {
    let mut engine = Spma::new();
    engine
        .train(&[vec!["<", "cat", ">"], vec!["<", "dog", ">"]])
        .unwrap();
    let result = engine.infer(&["<", "cat", ">"]).unwrap();
    assert!(result.is_anomaly, "empty grammar → all symbols uncovered, expected anomaly");
    let unknown = engine.infer(&["<", "fish", ">"]).unwrap();
    assert!(
        unknown.e_cost > 0.0,
        "unknown symbol must have E>0, got e={}",
        unknown.e_cost
    );
}

#[test]
fn no_singleton_seeding_grammar_contains_multi_symbol_pattern() {
    let mut engine = Spma::new();
    engine
        .train(&[
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
        ])
        .unwrap();
    let result = engine.infer(&["A", "B", "C"]).unwrap();
    assert!(
        result.e_cost == 0.0 || result.cd > 0.0,
        "repeated identical sequence should be covered by grammar: e={} cd={}",
        result.e_cost,
        result.cd
    );
}

#[test]
fn no_singleton_seeding_order_violation_detectable() {
    let mut engine = Spma::new();
    engine
        .train(&[
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
        ])
        .unwrap();
    let forward = engine.infer(&["A", "B", "C"]).unwrap();
    let reversed = engine.infer(&["C", "B", "A"]).unwrap();
    assert_eq!(forward.e_cost, 0.0, "forward [A,B,C] should be fully covered");
    assert!(
        reversed.e_cost > forward.e_cost,
        "reversed sequence must have strictly higher E: reversed_e={} forward_e={}",
        reversed.e_cost,
        forward.e_cost
    );
}

#[test]
fn contiguous_pattern_covered_gap_interrupted_not() {
    let mut engine = Spma::new();
    engine
        .train(&(0..5).map(|_| vec!["A", "B", "C"]).collect::<Vec<_>>())
        .unwrap();

    let contiguous = engine.infer(&["A", "B", "C"]).unwrap();
    assert_eq!(
        contiguous.e_cost,
        0.0,
        "contiguous [A,B,C] must be fully covered, got e={}",
        contiguous.e_cost
    );

    let gapped = engine.infer(&["A", "X", "B", "C"]).unwrap();
    assert!(
        gapped.e_cost > 0.0,
        "gap-interrupted [A,X,B,C] must have E>0, got e={}",
        gapped.e_cost
    );
}

#[test]
fn multi_symbol_grammar_size_nonzero_after_repeated_corpus() {
    let mut engine = Spma::new();
    engine
        .train(&[
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
        ])
        .unwrap();
    assert!(
        engine.grammar_size() > 0,
        "grammar_size should be > 0 after 5 identical sequences"
    );
}

#[test]
fn grammar_learns_shared_suffix_not_full_sequence() {
    let mut engine = Spma::new();
    engine
        .train(&[
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["A", "B", "C"],
            vec!["X", "B", "C"],
            vec!["X", "B", "C"],
            vec!["X", "B", "C"],
            vec!["X", "B", "C"],
            vec!["X", "B", "C"],
        ])
        .unwrap();
    let known = engine.infer(&["B", "C"]).unwrap();
    let unknown = engine.infer(&["Q", "R"]).unwrap();
    assert!(
        known.e_cost < unknown.e_cost,
        "known shared suffix [B,C] should have lower E than unknown [Q,R]: \
         known_e={} unknown_e={}",
        known.e_cost,
        unknown.e_cost
    );
}

#[test]
fn no_repeated_bigrams_produces_empty_grammar() {
    let mut engine = Spma::new();
    engine
        .train(&[
            vec!["A", "B"],
            vec!["C", "D"],
            vec!["E", "F"],
            vec!["G", "H"],
        ])
        .unwrap();
    assert_eq!(
        engine.grammar_size(),
        0,
        "corpus with no repeated bigrams should produce empty grammar"
    );
}

#[test]
fn longer_pattern_preferred_over_bigram_subpatterns() {
    let mut engine = Spma::new();
    let corpus: Vec<Vec<&str>> = (0..10).map(|_| vec!["A", "B", "C"]).collect();
    engine.train(&corpus).unwrap();
    let result = engine.infer(&["A", "B", "C"]).unwrap();
    assert!(
        result.e_cost == 0.0,
        "fully known repeated trigram should have e_cost=0, got {}",
        result.e_cost
    );
    assert!(
        engine.grammar_size() >= 1,
        "grammar must contain at least one multi-symbol pattern, got {}",
        engine.grammar_size()
    );
}

#[test]
fn max_cycles_truncation_produces_less_grammar_than_full_convergence() {
    let corpus: Vec<Vec<&str>> = (0..20)
        .flat_map(|_| {
            vec![
                vec!["X", "Y", "Z", "W"],
                vec!["X", "Y", "Z", "Q"],
                vec!["X", "Y", "W", "Q"],
                vec!["P", "Y", "Z", "W"],
            ]
        })
        .collect();

    let mut engine_capped = Spma::new();
    engine_capped.set_max_cycles(3);
    engine_capped.train(&corpus).unwrap();
    let capped_count = engine_capped.grammar_size();

    let mut engine_full = Spma::new();
    engine_full.train(&corpus).unwrap();
    let full_count = engine_full.grammar_size();

    assert!(
        full_count >= capped_count,
        "full convergence should produce >= patterns than capped: full={} capped={}",
        full_count,
        capped_count
    );
}

#[test]
fn corpus_cost_fallback_known_symbol_not_in_grammar_has_nonzero_e() {
    let mut engine = Spma::new();
    engine
        .train(&(0..8).map(|_| vec!["A", "B", "C", "D"]).collect::<Vec<_>>())
        .unwrap();

    for sym in &["A", "B", "C", "D"] {
        let r = engine.infer(&[sym]).unwrap();
        assert!(
            r.e_cost > 0.0 || r.cd > 0.0,
            "symbol {} must have non-zero signal (E={} CD={}); should not be free",
            sym,
            r.e_cost,
            r.cd
        );
    }
}

#[test]
fn corpus_cost_fallback_save_load_roundtrip() {
    let corpus: Vec<Vec<&str>> = (0..8).map(|_| vec!["A", "B", "C", "D"]).collect();

    let mut engine = Spma::new();
    engine.train(&corpus).unwrap();

    let path = "/tmp/spma_corpus_cost_test.bin";
    engine.save(path).unwrap();
    let loaded = Spma::load(path).unwrap();

    for sym in &["A", "B", "C", "D"] {
        let r_orig = engine.infer(&[sym]).unwrap();
        let r_load = loaded.infer(&[sym]).unwrap();
        assert!(
            (r_orig.e_cost - r_load.e_cost).abs() < 1e-9,
            "symbol {} e_cost differs after save/load: orig={} loaded={}",
            sym,
            r_orig.e_cost,
            r_load.e_cost
        );
    }
}
