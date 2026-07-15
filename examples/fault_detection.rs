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
            // Span contiguity is enforced per-pattern, but multiple grammar patterns can
            // each contribute contiguous spans that together cover a reordered input.
            // A single pattern [TRIP_A, BREAKER_OPEN] cannot scatter-match, but
            // [TRIP_B BREAKER_OPEN] covers pos 0 and [TRIP_A BREAKER_OPEN] covers pos 1 —
            // two valid contiguous matches, wrong order still scores E=0.
            "OK (order violation undetected — multi-pattern stitching covers reordered input)",
            vec!["BREAKER_OPEN", "TRIP_A", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Anomaly — unknown fault type",
            vec!["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"],
        ),
        (
            // All 3 symbols appear in grammar patterns → E=0, not detected.
            // Sequence-length anomalies require boundary markers (< >) to be detectable.
            "False negative: missing relay (not detected — all symbols individually known)",
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
