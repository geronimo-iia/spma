use spma::Spma;

#[test]
fn gap_patterns_dont_break_contiguous_inference() {
    // Step 1: train WITHOUT gap induction
    let mut spma_contiguous = Spma::new(5);
    spma_contiguous.set_max_induced_gap(0);
    spma_contiguous.train(&vec![vec!["A", "B", "C"]; 8]);
    let result_contiguous = spma_contiguous.infer(&["A", "B", "C"]);

    // Step 2: train WITH gap induction (default max_induced_gap=3)
    let mut spma_gap = Spma::new(5);
    spma_gap.train(&vec![vec!["A", "B", "C"]; 8]);
    let result_gap = spma_gap.infer(&["A", "B", "C"]);

    assert!(
        result_contiguous.e_norm < 1e-10,
        "contiguous-only: fully covered sequence must have e_norm ~0, got {}",
        result_contiguous.e_norm
    );
    assert!(
        result_gap.e_norm < 1e-10,
        "gap-enabled: fully covered sequence must have e_norm ~0, got {}",
        result_gap.e_norm
    );
    assert!(
        !result_contiguous.is_anomaly,
        "contiguous-only: known sequence must not be anomaly"
    );
    assert!(
        !result_gap.is_anomaly,
        "gap-enabled: known sequence must not be anomaly"
    );
    assert!(
        result_gap.alignment.rows.len() >= 1,
        "gap-enabled: alignment must have at least 1 row, got {}",
        result_gap.alignment.rows.len()
    );

    // Verify contiguous match log intact with gap training enabled
    let mut spma_gap2 = Spma::new(5);
    spma_gap2.train(&vec![vec!["A", "B", "C"]; 8]);
    let result2 = spma_gap2.infer(&["A", "B", "C"]);
    assert_eq!(
        result2.alignment.covered,
        vec![true, true, true],
        "gap-enabled: all symbols must be covered, got {:?}",
        result2.alignment.covered
    );
}

#[test]
fn infer_same_sequence_twice_identical_results() {
    let mut spma = Spma::new(5);
    spma.train(&vec![vec!["A", "B", "C", "D"]; 8]);

    let r1 = spma.infer(&["A", "B", "C", "D"]);
    let r2 = spma.infer(&["A", "B", "C", "D"]);

    assert_eq!(r1.e_cost, r2.e_cost, "e_cost must be deterministic");
    assert_eq!(r1.e_norm, r2.e_norm, "e_norm must be deterministic");
    assert_eq!(r1.is_anomaly, r2.is_anomaly, "is_anomaly must be deterministic");
    assert_eq!(r1.cd, r2.cd, "cd must be deterministic");
    assert_eq!(
        r1.alignment.covered, r2.alignment.covered,
        "alignment.covered must be deterministic"
    );
    assert_eq!(
        r1.alignment.rows.len(),
        r2.alignment.rows.len(),
        "alignment.rows.len() must be deterministic"
    );

    // Determinism on unknown sequence
    let u1 = spma.infer(&["X", "Y", "Z"]);
    let u2 = spma.infer(&["X", "Y", "Z"]);

    assert_eq!(u1.e_cost, u2.e_cost, "unknown: e_cost must be deterministic");
    assert_eq!(u1.e_norm, u2.e_norm, "unknown: e_norm must be deterministic");
}
