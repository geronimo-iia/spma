//! Demonstrates training a grammar on normal industrial event sequences
//! and running anomaly detection. (Save/load deferred to Phase 3.)

use spma::Spma;

fn main() {
    let normal: Vec<Vec<&str>> = vec![
        vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_A", "BREAKER_OPEN", "OVERCURRENT",  "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT",  "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT",  "BACKUP_RELAY"],
        vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
    ];

    let mut engine = Spma::new(10);
    engine.train(&normal);

    let test_cases: Vec<(&str, Vec<&str>)> = vec![
        ("Normal sequence",
            vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"]),
        ("Normal — variant",
            vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"]),
        ("Reordered (may not be detected with varied corpus)",
            vec!["BREAKER_OPEN", "TRIP_A", "UNDERVOLTAGE", "BACKUP_RELAY"]),
        ("Anomaly — unknown fault type",
            vec!["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"]),
        ("Anomaly — completely novel",
            vec!["SENSOR_FAIL", "WATCHDOG_RESET", "REBOOT"]),
    ];

    for (label, seq) in &test_cases {
        let result = engine.infer(seq);
        let tag = if result.is_anomaly { "ANOMALY" } else { "OK     " };
        println!(
            "[{tag}]  E={:.3}  CD={:+.3}  — {label}",
            result.e_cost, result.cd
        );
        let unmatched = result.alignment.unmatched_symbols();
        if result.is_anomaly && !unmatched.is_empty() {
            println!("         unmatched: {}", unmatched.join(", "));
        }
        println!("{}", result.alignment);
    }
}
