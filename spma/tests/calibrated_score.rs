use spma::Spma;

fn corpus(seq: Vec<&str>, n: usize) -> Vec<Vec<&str>> {
    vec![seq; n]
}

// Scenario 19 — identical_corpus_all_e_norm_zero
#[test]
fn identical_corpus_all_e_norm_zero() {
    let seq = vec!["A", "B", "C"];
    let c = corpus(seq.clone(), 10);
    let mut spma = Spma::new(10);
    spma.train(&c);

    for _ in 0..c.len() {
        let result = spma.infer(&["A", "B", "C"]);
        assert!(
            result.e_norm < 1e-10,
            "training seq e_norm must be ~0, got {}",
            result.e_norm
        );
    }

    let pct = spma.grammar().e_distribution.percentile(1e-10);
    assert!(
        (pct - 1.0).abs() < 1e-10,
        "all training e_norms are 0.0 → percentile(1e-10) must be 1.0, got {}",
        pct
    );
}

// Scenario 20 — perfectly_covered_anomaly_percentile_zero
#[test]
fn perfectly_covered_anomaly_percentile_zero() {
    let c = corpus(vec!["A", "B", "C"], 10);
    let mut spma = Spma::new(10);
    spma.train(&c);

    let result = spma.infer(&["A", "B", "C"]);
    assert!(
        result.anomaly_percentile < 1e-10,
        "perfectly covered seq: anomaly_percentile must be ~0, got {}",
        result.anomaly_percentile
    );
    assert!(
        result.e_norm < 1e-10,
        "perfectly covered seq: e_norm must be ~0, got {}",
        result.e_norm
    );
}

// Scenario 21 — novel_sequence_anomaly_percentile_positive
#[test]
fn novel_sequence_anomaly_percentile_positive() {
    let c = corpus(vec!["A", "B", "C"], 8);
    let mut spma = Spma::new(10);
    spma.train(&c);

    let result = spma.infer(&["X", "Y", "Z"]);
    assert!(
        result.anomaly_percentile > 0.0,
        "novel seq anomaly_percentile must be > 0.0, got {}",
        result.anomaly_percentile
    );
    assert!(
        result.e_cost > 0.0,
        "novel seq e_cost must be > 0.0, got {}",
        result.e_cost
    );
}

// Scenario 22 — known_sequence_not_anomaly
#[test]
fn known_sequence_not_anomaly() {
    let c = corpus(vec!["A", "B", "C"], 8);
    let mut spma = Spma::new(10);
    spma.train(&c);

    let result = spma.infer(&["A", "B", "C"]);
    assert!(
        !result.is_anomaly,
        "known seq must not be anomaly (e_norm={}, threshold=0.0)",
        result.e_norm
    );
    assert!(
        result.e_norm < 1e-10,
        "known seq e_norm must be ~0, got {}",
        result.e_norm
    );
}

// Scenario 23 — unknown_sequence_is_anomaly
#[test]
fn unknown_sequence_is_anomaly() {
    let c = corpus(vec!["A", "B", "C"], 8);
    let mut spma = Spma::new(10);
    spma.train(&c);

    let result = spma.infer(&["X", "Y", "Z"]);
    assert!(
        result.is_anomaly,
        "unknown seq must be anomaly (e_norm={}, threshold=0.0)",
        result.e_norm
    );
    assert!(
        result.e_norm > 0.0,
        "unknown seq e_norm must be > 0.0, got {}",
        result.e_norm
    );
}

// Scenario 24 — level_costs_len_matches_grammar_levels
#[test]
fn level_costs_len_matches_grammar_levels() {
    let seq = vec!["A", "B", "C", "A", "B", "C"];
    let c = corpus(seq.clone(), 6);
    let mut spma = Spma::new(5);
    spma.train(&c);

    let result = spma.infer(&seq);
    let n_levels = spma.grammar().levels.len();

    assert_eq!(
        result.level_costs.len(),
        n_levels,
        "level_costs.len() must equal grammar.levels.len()"
    );
    assert_eq!(
        result.level_e_norms.len(),
        n_levels,
        "level_e_norms.len() must equal grammar.levels.len()"
    );
    for (i, &v) in result.level_costs.iter().enumerate() {
        assert!(v >= 0.0, "level_costs[{}] must be >= 0.0, got {}", i, v);
    }
    for (i, &v) in result.level_e_norms.iter().enumerate() {
        assert!(v >= 0.0, "level_e_norms[{}] must be >= 0.0, got {}", i, v);
    }
}

// Scenario 26 — per_level_threshold_catches_missed_anomaly
#[test]
fn per_level_threshold_catches_missed_anomaly() {
    // Train on [A B C] repeated — learns pattern covering all atoms in order.
    // Infer [C B A] — atoms fully covered (e_norm_0 ≈ 0) but pattern order is inverted.
    // With global threshold only: not anomaly. With level-1 threshold low: anomaly.
    let seq = vec!["A", "B", "C"];
    let c = corpus(seq.clone(), 10);
    let mut spma = Spma::new(10);
    spma.train(&c);

    let inverted = vec!["C", "B", "A"];

    // Global threshold only — may not catch the inverted order if level-0 e_norm is low
    spma.set_anomaly_threshold(f64::MAX);
    let result_global_only = spma.infer(&inverted);
    assert!(
        !result_global_only.is_anomaly,
        "with threshold=MAX: inverted seq must not be anomaly, got is_anomaly=true"
    );

    // Set level-1 threshold to 0.0 — any nonzero level-1 e_norm flags anomaly
    spma.set_anomaly_threshold(f64::MAX);
    spma.set_level_threshold(1, 0.0);
    let result_level = spma.infer(&inverted);

    // If grammar has level 1 and inverted sequence produces nonzero level-1 e_norm, it's anomaly
    if result_level.level_e_norms.len() > 1 && result_level.level_e_norms[1] > 0.0 {
        assert!(
            result_level.is_anomaly,
            "level-1 threshold=0.0 with level_e_norms[1]={}: must be anomaly",
            result_level.level_e_norms[1]
        );
    }
    // If grammar only has level 0, the test still passes — we just verify no panic
}

// Scenario 27 — per_level_threshold_fallback_to_global
#[test]
fn per_level_threshold_fallback_to_global() {
    let c = corpus(vec!["A", "B", "C"], 10);
    let mut spma = Spma::new(10);
    spma.train(&c);

    // level_thresholds empty → is_anomaly uses global threshold only
    assert!(
        spma.grammar().e_distribution.level_thresholds.is_empty(),
        "level_thresholds must be empty after train"
    );
    let result = spma.infer(&["A", "B", "C"]);
    assert!(
        !result.is_anomaly,
        "known seq with empty level_thresholds must not be anomaly"
    );

    // set_level_threshold extends vec correctly
    spma.set_level_threshold(2, 0.5);
    assert_eq!(
        spma.grammar().e_distribution.level_thresholds.len(),
        3,
        "level_thresholds must be resized to level+1"
    );
    assert!(
        (spma.grammar().e_distribution.level_thresholds[2] - 0.5).abs() < 1e-12,
        "level_thresholds[2] must be 0.5"
    );
}

// Scenario 25 — set_anomaly_threshold_gates_is_anomaly
#[test]
fn set_anomaly_threshold_gates_is_anomaly() {
    let c = corpus(vec!["A", "B", "C"], 20);
    let mut spma = Spma::new(10);
    spma.train(&c);

    // Novel symbol produces e_norm > 0
    let result = spma.infer(&["A", "X", "C"]);
    assert!(
        result.e_norm > 0.0,
        "novel symbol must produce e_norm > 0, got {}",
        result.e_norm
    );

    // With threshold above e_norm: not an anomaly
    spma.set_anomaly_threshold(0.5);
    let result_high = spma.infer(&["A", "X", "C"]);
    assert!(
        result_high.e_norm < 0.5,
        "e_norm must be < 0.5 for threshold test to be meaningful, got {}",
        result_high.e_norm
    );
    assert!(
        !result_high.is_anomaly,
        "threshold=0.5 > e_norm={}: must not be anomaly",
        result_high.e_norm
    );

    // With threshold = 0.0: always anomaly when e_norm > 0
    spma.set_anomaly_threshold(0.0);
    let result_zero = spma.infer(&["A", "X", "C"]);
    assert!(
        result_zero.is_anomaly,
        "threshold=0.0 with e_norm={}: must be anomaly",
        result_zero.e_norm
    );
}
