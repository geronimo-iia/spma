//! Demonstrates training a grammar on normal industrial event sequences
//! and running anomaly detection. (Save/load deferred to Phase 3.)
//!
//! # Expected output
//!
//! ```text
//! [OK     ]  E=-0.000  CD=+9.678  — Normal sequence
//!         TRIP_A        BREAKER_OPEN  UNDERVOLTAGE  BACKUP_RELAY
//! P2(L0)  TRIP_A        .             .             .
//! P1(L0)  .             BREAKER_OPEN  UNDERVOLTAGE  BACKUP_RELAY
//! ---
//! E: -0.0 bits   CD: 9.7 bits   T: -0.0 bits
//! [ANOMALY]  E=3.415  CD=+7.000  — Normal — variant
//!          unmatched: OVERCURRENT
//!         TRIP_B        BREAKER_OPEN  OVERCURRENT   BACKUP_RELAY
//! P3(L0)  TRIP_B        .             .             .
//! P0(L0)  .             BREAKER_OPEN  <1>           BACKUP_RELAY
//! ---
//! E: 3.4 bits   CD: 7.0 bits   T: 3.4 bits
//! [ANOMALY]  E=4.678  CD=+5.000  — Reordered (may not be detected with varied corpus)
//!          unmatched: UNDERVOLTAGE, BACKUP_RELAY
//!         BREAKER_OPEN  TRIP_A        UNDERVOLTAGE  BACKUP_RELAY
//! P0(L0)  BREAKER_OPEN  .             .             .
//! P2(L0)  .             TRIP_A        .             .
//! ---
//! E: 4.7 bits   CD: 5.0 bits   T: 4.7 bits
//! [ANOMALY]  E=3.415  CD=+7.000  — Anomaly — unknown fault type
//!          unmatched: GROUNDFAULT
//!         TRIP_A        BREAKER_OPEN  GROUNDFAULT   BACKUP_RELAY
//! P2(L0)  TRIP_A        .             .             .
//! P0(L0)  .             BREAKER_OPEN  <1>           BACKUP_RELAY
//! ---
//! E: 3.4 bits   CD: 7.0 bits   T: 3.4 bits
//! [ANOMALY]  E=10.245  CD=+0.000  — Anomaly — completely novel
//!          unmatched: SENSOR_FAIL, WATCHDOG_RESET, REBOOT
//!   SENSOR_FAIL     WATCHDOG_RESET  REBOOT
//! ---
//! E: 10.2 bits   CD: 0.0 bits   T: 10.2 bits
//! ```

use spma::Spma;

fn main() {
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

    let mut engine = Spma::new(10);
    engine.train(&normal);

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
            "Reordered (may not be detected with varied corpus)",
            vec!["BREAKER_OPEN", "TRIP_A", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Anomaly — unknown fault type",
            vec!["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"],
        ),
        (
            "Anomaly — completely novel",
            vec!["SENSOR_FAIL", "WATCHDOG_RESET", "REBOOT"],
        ),
    ];

    for (label, seq) in &test_cases {
        let result = engine.infer(seq);
        let tag = if result.is_anomaly {
            "ANOMALY"
        } else {
            "OK     "
        };
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
