//! Demonstrates training a grammar on normal industrial event sequences,
//! saving it, loading it, and running anomaly detection.

use spma::Spma;

fn main() -> anyhow::Result<()> {
    // --- Training corpus: normal fault-handling sequences ---
    let normal: Vec<Vec<&str>> = vec![
        vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_A", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
        vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
    ];

    let grammar_path = "/tmp/spma_fault_demo.bin";

    // Train and persist
    let mut engine = Spma::new();
    engine.train(&normal)?;
    engine.save(grammar_path)?;
    println!("Grammar saved to {grammar_path}\n");

    // Load from disk (simulates a separate inference process)
    let engine = Spma::load(grammar_path)?;

    let test_cases: Vec<(&str, Vec<&str>)> = vec![
        (
            "Normal sequence",
            vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Normal — variant",
            vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
        ),
        (
            // Beam search matches symbols by identity, not position — order violations
            // are NOT detected unless the grammar contains ordered multi-symbol patterns.
            "OK (order violation undetected — beam is order-agnostic)",
            vec!["BREAKER_OPEN", "TRIP_A", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Anomaly — unknown fault type",
            vec!["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"],
        ),
        (
            "Anomaly — missing relay",
            vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE"],
        ),
        (
            "Anomaly — completely novel",
            vec!["SENSOR_FAIL", "WATCHDOG_RESET", "REBOOT"],
        ),
    ];

    for (label, seq) in &test_cases {
        let result = engine.infer(seq)?;
        let tag = if result.is_anomaly { "ANOMALY" } else { "OK     " };
        println!(
            "[{tag}]  E={:.3}  CD={:+.3}  — {label}",
            result.e_cost, result.cd
        );
        if result.is_anomaly && !result.unmatched.is_empty() {
            println!("         unmatched: {}", result.unmatched.join(", "));
        }
        println!("{}", result.alignment);
    }

    Ok(())
}
