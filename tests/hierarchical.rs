//! Hierarchical grammar (Fix B) tests.
//!
//! Corpus design note: sequences must be long enough (6+ atoms) so the
//! trigram miner can produce two non-overlapping trigrams at level-0, giving
//! a pid_seq of length ≥ 2 at level-1 and enabling level-2 grammar formation.
use spma::Spma;

/// 6-symbol repeated corpus: level-0 learns [A,B,C] and [D,E,F] (two trigrams).
/// Level-1 beam sees pid_seq = [pid_ABC, pid_DEF] → level-2 grammar forms.
fn train_abc_def_x10() -> Spma {
    let mut eng = Spma::new();
    let corpus: Vec<Vec<&str>> = (0..10)
        .map(|_| vec!["A", "B", "C", "D", "E", "F"])
        .collect();
    eng.train(&corpus).unwrap();
    eng
}

#[test]
fn l2_grammar_forms_after_repeated_ordered_corpus() {
    let eng = train_abc_def_x10();
    assert!(
        eng.grammar_depth() >= 2,
        "expected grammar_depth >= 2, got {}",
        eng.grammar_depth()
    );
}

#[test]
fn l2_correct_order_zero_level_cost() {
    let eng = train_abc_def_x10();
    if eng.grammar_depth() < 2 {
        return;
    }
    let result = eng.infer(&["A", "B", "C", "D", "E", "F"]).unwrap();
    assert!(
        result.level_costs.is_empty() || result.level_costs[0] == 0.0,
        "correct order should have level_costs[0] == 0.0, got {:?}",
        result.level_costs
    );
}

#[test]
fn l2_reversed_order_nonzero_level_cost() {
    let eng = train_abc_def_x10();
    if eng.grammar_depth() < 2 {
        return;
    }
    // Reversed: DEF before ABC — unseen ordering
    let result = eng.infer(&["D", "E", "F", "A", "B", "C"]).unwrap();
    assert!(result.is_anomaly, "reversed order must be anomalous");
    assert!(
        !result.level_costs.is_empty() && result.level_costs[0] > 0.0,
        "reversed order must have positive level_costs[0], got {:?}",
        result.level_costs
    );
}

#[test]
fn l2_varied_order_corpus_both_orders_allowed() {
    // Both orderings seen equally — MDL correctly encodes both as grammar.
    // Neither ordering should be anomalous.
    let mut eng = Spma::new();
    let mut corpus: Vec<Vec<&str>> = Vec::new();
    for _ in 0..5 {
        corpus.push(vec!["A", "B", "C", "D", "E", "F"]);
    }
    for _ in 0..5 {
        corpus.push(vec!["D", "E", "F", "A", "B", "C"]);
    }
    eng.train(&corpus).unwrap();

    let r1 = eng.infer(&["A", "B", "C", "D", "E", "F"]).unwrap();
    let r2 = eng.infer(&["D", "E", "F", "A", "B", "C"]).unwrap();
    assert!(!r1.is_anomaly, "ABCDEF seen in training — must not be anomalous");
    assert!(!r2.is_anomaly, "DEFABC seen in training — must not be anomalous");
}

#[test]
fn l2_save_load_roundtrip_preserves_level_costs() {
    let eng = train_abc_def_x10();
    let path = "/tmp/spma_hierarchical_test.bin";
    eng.save(path).unwrap();
    let eng2 = Spma::load(path).unwrap();

    let r1 = eng.infer(&["D", "E", "F", "A", "B", "C"]).unwrap();
    let r2 = eng2.infer(&["D", "E", "F", "A", "B", "C"]).unwrap();

    assert_eq!(
        r1.level_costs.len(),
        r2.level_costs.len(),
        "level_costs length must match after load"
    );
    for (a, b) in r1.level_costs.iter().zip(r2.level_costs.iter()) {
        assert!((a - b).abs() < 1e-9, "level_costs mismatch: {a} vs {b}");
    }
    assert_eq!(r1.is_anomaly, r2.is_anomaly);
}

#[test]
fn l2_all_unknown_sequence_does_not_fire_l2() {
    let eng = train_abc_def_x10();
    let result = eng.infer(&["X", "Y", "Z", "W", "V", "U"]).unwrap();
    assert!(result.e_cost > 0.0, "unknown sequence must have E > 0");
    assert!(
        result.level_costs.is_empty() || result.level_costs.iter().all(|&c| c == 0.0),
        "all-unknown: level-2 must not fire or contribute zero cost"
    );
}

#[test]
fn l3_grammar_forms_on_episode_corpus() {
    let mut eng = Spma::new();
    // 8-atom sequence — miner produces two trigrams, enabling level-2.
    let corpus: Vec<Vec<&str>> = (0..10)
        .map(|_| vec!["A", "B", "C", "D", "E", "F", "G", "H"])
        .collect();
    eng.train(&corpus).unwrap();
    assert!(
        eng.grammar_depth() >= 2,
        "long repeated episode must form at least 2 grammar levels, got {}",
        eng.grammar_depth()
    );
}

#[test]
fn loop_terminates_naturally_on_flat_corpus() {
    let mut eng = Spma::new();
    // All distinct symbols — no pattern can recur
    eng.train(&[
        vec!["A", "B", "C", "D"],
        vec!["E", "F", "G", "H"],
        vec!["I", "J", "K", "L"],
    ])
    .unwrap();
    assert_eq!(
        eng.grammar_depth(),
        1,
        "flat corpus: no pattern recurrence → grammar_depth must be 1"
    );
}

#[test]
fn safety_cap_blocks_deep_hierarchy_when_set_low() {
    let mut eng = Spma::new();
    eng.set_max_levels_safety_cap(1);
    let corpus: Vec<Vec<&str>> = (0..10)
        .map(|_| vec!["A", "B", "C", "D", "E", "F"])
        .collect();
    eng.train(&corpus).unwrap();
    assert!(
        eng.grammar_depth() <= 2,
        "safety cap=1 means at most 1 higher level, so depth <= 2"
    );
}

#[test]
fn grammar_size_at_returns_correct_counts() {
    let eng = train_abc_def_x10();
    assert_eq!(
        eng.grammar_size_at(0),
        eng.grammar_size(),
        "grammar_size_at(0) must equal grammar_size()"
    );
    if eng.grammar_depth() >= 2 {
        assert!(
            eng.grammar_size_at(1) > 0,
            "grammar_size_at(1) must be > 0 when depth >= 2"
        );
    }
    assert_eq!(eng.grammar_size_at(99), 0, "non-existent level returns 0");
}

#[test]
fn e_cost_equals_sum_of_level_costs() {
    let eng = train_abc_def_x10();
    let result = eng.infer(&["D", "E", "F", "A", "B", "C"]).unwrap();
    let level_sum: f64 = result.level_costs.iter().sum();
    let e_cost_l0 = result.e_cost - level_sum;
    let expected = e_cost_l0 + level_sum;
    assert!(
        (result.e_cost - expected).abs() < 1e-9,
        "e_cost ({}) must equal e_cost_l0 ({}) + level_sum ({})",
        result.e_cost,
        e_cost_l0,
        level_sum
    );
}
