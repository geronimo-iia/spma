//! Demonstrates order-sensitive anomaly detection and its current limits.
//!
//! When the training corpus is a single repeated sequence, the grammar learns
//! tight multi-symbol patterns (e.g. [TRIP_A, BREAKER_OPEN, UNDERVOLTAGE]).
//! A reordered input cannot match these patterns contiguously → E > 0.
//!
//! This works when:
//!   - The corpus is homogeneous (one canonical sequence, repeated many times).
//!   - The grammar therefore learns patterns that span the full sequence.
//!
//! This does NOT work when:
//!   - Multiple sequence variants are in the corpus — grammar learns shorter,
//!     more flexible patterns that can stitch across reorderings.

use spma::Spma;

fn main() {
    // ── Homogeneous corpus: order detection works ────────────────────────────

    println!("=== Homogeneous corpus (single sequence type × 8) ===\n");

    let mut engine = Spma::new(10);
    engine.train(
        &(0..8)
            .map(|_| vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"])
            .collect::<Vec<_>>(),
    );

    let cases: Vec<(&str, Vec<&str>)> = vec![
        (
            "Normal",
            vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Reordered — detected (grammar spans full sequence)",
            vec!["BACKUP_RELAY", "UNDERVOLTAGE", "BREAKER_OPEN", "TRIP_A"],
        ),
        (
            "Unknown symbol — detected",
            vec!["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"],
        ),
        (
            "Missing symbol — NOT detected (remaining symbols still covered)",
            vec!["TRIP_A", "BREAKER_OPEN", "BACKUP_RELAY"],
        ),
    ];

    for (label, seq) in &cases {
        let r = engine.infer(seq);
        let tag = if r.is_anomaly { "ANOMALY" } else { "OK     " };
        println!("[{tag}]  E={:.3}  CD={:+.3}  — {label}", r.e_cost, r.cd);
        let unmatched = r.alignment.unmatched_symbols();
        if !unmatched.is_empty() {
            println!("         unmatched: {}", unmatched.join(", "));
        }
    }

    // ── Varied corpus: order detection breaks down ───────────────────────────

    println!("\n=== Varied corpus (4 variants × 8) ===\n");
    println!("Grammar learns shorter patterns that stitch across reorderings.\n");

    let mut engine2 = Spma::new(10);
    engine2.train(
        &(0..8)
            .flat_map(|_| {
                vec![
                    vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
                    vec!["TRIP_A", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
                    vec!["TRIP_B", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
                    vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
                ]
            })
            .collect::<Vec<_>>(),
    );

    let cases2: Vec<(&str, Vec<&str>)> = vec![
        (
            "Normal",
            vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Reordered — partially detected (multi-pattern stitching reduces E)",
            vec!["BACKUP_RELAY", "UNDERVOLTAGE", "BREAKER_OPEN", "TRIP_A"],
        ),
        (
            "Unknown symbol — detected",
            vec!["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"],
        ),
    ];

    for (label, seq) in &cases2 {
        let r = engine2.infer(seq);
        let tag = if r.is_anomaly { "ANOMALY" } else { "OK     " };
        println!("[{tag}]  E={:.3}  CD={:+.3}  — {label}", r.e_cost, r.cd);
        let unmatched = r.alignment.unmatched_symbols();
        if !unmatched.is_empty() {
            println!("         unmatched: {}", unmatched.join(", "));
        }
    }

    println!("\nSummary:");
    println!("  - Unknown symbols:  always detected");
    println!("  - Order violations: detected with homogeneous corpus; weaker with varied");
    println!("  - Missing symbols:  not detected (remaining symbols still covered)");
}
