use spma::Spma;

// Scenario 28
#[test]
fn empty_sequence_no_panic() {
    let mut spma = Spma::new(5);
    spma.train(&vec![vec!["A", "B", "C"]; 6]);
    let result = spma.infer(&[]);
    assert_eq!(result.e_cost, 0.0);
    assert_eq!(result.e_norm, 0.0);
    assert!(result.alignment.new_symbols.is_empty());
    assert!(result.alignment.covered.is_empty());
}

// Scenario 29
#[test]
fn single_symbol_sequence_no_panic() {
    let mut spma = Spma::new(5);
    spma.train(&vec![vec!["A", "B", "C"]; 6]);
    let result = spma.infer(&["A"]);
    assert_eq!(result.alignment.new_symbols.len(), 1);
    assert_eq!(result.alignment.covered.len(), 1);
    assert!(result.e_cost >= 0.0);
}

// Scenario 30
#[test]
fn unknown_symbol_e_cost_positive_no_panic() {
    let mut spma = Spma::new(5);
    spma.train(&vec![vec!["A", "B", "C"]; 6]);
    let result = spma.infer(&["UNKNOWN_XYZ_QQQ"]);
    assert!(
        result.e_cost > 0.0,
        "unknown symbol must have positive e_cost, got {}",
        result.e_cost
    );
    assert!(
        result.is_anomaly,
        "unknown symbol must be flagged as anomaly"
    );
}

// Scenario 31
#[test]
fn single_training_sequence_no_panic() {
    let mut spma = Spma::new(5);
    // Only 1 sequence — below min_freq threshold (max(1/2, 2) = 2)
    spma.train(&[vec!["A", "B", "C"]]);
    let result = spma.infer(&["A", "B", "C"]);
    assert!(
        result.e_cost >= 0.0,
        "e_cost must be non-negative, got {}",
        result.e_cost
    );
    assert_eq!(result.alignment.new_symbols.len(), 3);
}

// Scenario 32
#[test]
fn beam_k_one_returns_result() {
    let mut spma = Spma::new(1);
    spma.train(&vec![vec!["A", "B", "C"]; 6]);
    let result = spma.infer(&["A", "B", "C"]);
    assert!(
        result.e_cost >= 0.0,
        "e_cost must be non-negative, got {}",
        result.e_cost
    );
    assert_eq!(result.alignment.new_symbols.len(), 3);

    let mut spma2 = Spma::new(1);
    spma2.train(&vec![vec!["A", "B", "C", "D", "E"]; 8]);
    let result2 = spma2.infer(&["A", "B", "C", "D", "E"]);
    assert!(
        result2.e_cost >= 0.0,
        "e_cost must be non-negative, got {}",
        result2.e_cost
    );
}
