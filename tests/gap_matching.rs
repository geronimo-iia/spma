use spma::model::SymbolRef;
use spma::Spma;

fn trip_restoration_corpus() -> Vec<Vec<&'static str>> {
    vec![
        vec!["TRIP", "OVERCURRENT",  "RESTORATION"],
        vec!["TRIP", "UNDERVOLTAGE", "RESTORATION"],
        vec!["TRIP", "EARTH_FAULT",  "RESTORATION"],
        vec!["TRIP", "PHASE_FAULT",  "RESTORATION"],
        vec!["TRIP", "OVERCURRENT",  "RESTORATION"],
        vec!["TRIP", "UNDERVOLTAGE", "RESTORATION"],
        vec!["TRIP", "EARTH_FAULT",  "RESTORATION"],
        vec!["TRIP", "PHASE_FAULT",  "RESTORATION"],
    ]
}

#[test]
fn gap_pattern_learned_and_matches_novel_middle() {
    let mut spma = Spma::new(10);
    spma.set_max_induced_gap(1);
    spma.train(&trip_restoration_corpus());

    let result = spma.infer(&["TRIP", "NEW_FAULT", "RESTORATION"]);
    assert!(
        result.e_norm < 1.0,
        "TRIP+RESTORATION gap pattern must compress novel middle: e_norm={} >= 1.0",
        result.e_norm
    );

    let trip_id = spma.grammar.interner.get("TRIP").expect("TRIP must be interned");
    let rest_id = spma.grammar.interner.get("RESTORATION").expect("RESTORATION must be interned");
    let has_trip_restoration_pattern = spma.grammar.levels[0].patterns.iter().any(|p| {
        let has_trip = p.symbols.iter().any(|s| matches!(s, SymbolRef::Atom(id) if *id == trip_id));
        let has_rest = p.symbols.iter().any(|s| matches!(s, SymbolRef::Atom(id) if *id == rest_id));
        has_trip && has_rest
    });
    assert!(
        has_trip_restoration_pattern,
        "grammar must contain a pattern covering both TRIP and RESTORATION"
    );
}

#[test]
fn gap_too_wide_not_covered() {
    let mut spma = Spma::new(10);
    spma.set_max_induced_gap(1);
    spma.train(&trip_restoration_corpus());

    let result_wide   = spma.infer(&["TRIP", "A", "B", "RESTORATION"]);
    let result_narrow = spma.infer(&["TRIP", "X", "RESTORATION"]);

    assert!(
        result_wide.e_cost > result_narrow.e_cost,
        "2-gap sequence must cost more than 1-gap: wide={} narrow={}",
        result_wide.e_cost,
        result_narrow.e_cost
    );
}

#[test]
fn wrong_order_is_anomaly() {
    let mut spma = Spma::new(10);
    spma.set_max_induced_gap(1);
    spma.train(&trip_restoration_corpus());

    let result = spma.infer(&["RESTORATION", "TRIP"]);
    assert!(
        result.is_anomaly,
        "reversed order must be anomaly"
    );
    assert!(
        result.e_norm > 0.0,
        "reversed order must have e_norm > 0.0, got {}",
        result.e_norm
    );

}
