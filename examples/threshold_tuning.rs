//! Demonstrates threshold tuning using the training E_norm distribution.
//!
//! Default threshold is 0.0 (any uncovered symbol = anomaly). Inspecting the
//! training distribution lets you pick a threshold that tolerates partial
//! coverage — useful when normal sequences sometimes contain rare atoms.

use spma::Spma;

fn main() {
    // Varied corpus: some atoms will not be fully covered by learned patterns.
    let corpus: Vec<Vec<&str>> = (0..8)
        .flat_map(|_| {
            vec![
                vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
                vec!["TRIP_A", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
                vec!["TRIP_B", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
                vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
            ]
        })
        .collect();

    let mut model = Spma::new(10);
    model.train(&corpus);

    // Inspect the training E_norm distribution.
    let dist = model.e_distribution();
    println!("Training E_norm distribution:");
    println!("  p50  = {:.3}", dist.quantile(0.5));
    println!("  p90  = {:.3}", dist.quantile(0.9));
    println!("  p95  = {:.3}", dist.quantile(0.95));
    println!("  p99  = {:.3}", dist.quantile(0.99));
    println!("  current threshold = {:.3}", dist.threshold);

    let test_cases: Vec<(&str, Vec<&str>)> = vec![
        (
            "Normal sequence",
            vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Unknown atom",
            vec!["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"],
        ),
        (
            "Completely novel",
            vec!["SENSOR_FAIL", "WATCHDOG_RESET", "REBOOT"],
        ),
    ];

    // Compare default threshold (0.0) vs p90.
    let p90 = dist.quantile(0.9);

    println!("\n--- threshold=0.0 (default) ---");
    for (label, seq) in &test_cases {
        let r = model.infer(seq);
        let tag = if r.is_anomaly { "ANOMALY" } else { "OK     " };
        println!("[{tag}]  e_norm={:.3}  — {label}", r.e_norm);
    }

    model.set_anomaly_threshold(p90);

    println!("\n--- threshold={p90:.3} (p90 of training distribution) ---");
    for (label, seq) in &test_cases {
        let r = model.infer(seq);
        let tag = if r.is_anomaly { "ANOMALY" } else { "OK     " };
        println!("[{tag}]  e_norm={:.3}  — {label}", r.e_norm);
    }

    println!("\nNote: raising the threshold reduces false positives on sequences");
    println!("with partial coverage, at the cost of missing low-e_norm anomalies.");
}
