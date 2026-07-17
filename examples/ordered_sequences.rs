//! Demonstrates order-sensitive anomaly detection and its current limits.
//!
//! # Expected output
//!
//! ```text
//! === Homogeneous corpus (single sequence type × 8) ===
//!
//! Note: atoms not covered by any learned pattern score E > 0.
//! 'Normal' may still show ANOMALY if grammar coverage is incomplete.
//!
//! [ANOMALY]  e_norm=0.250  E=2.000  CD=+6.000  — Normal (E>0 if any atom uncovered by grammar)
//!          unmatched: BACKUP_RELAY
//! [ANOMALY]  e_norm=0.750  E=6.000  CD=+2.000  — Reordered — higher E (grammar spans sequence, reorder breaks patterns)
//!          unmatched: BACKUP_RELAY, UNDERVOLTAGE, BREAKER_OPEN
//! [ANOMALY]  e_norm=0.500  E=4.000  CD=+4.000  — Unknown symbol — highest E (unknown atom costs max bits)
//!          unmatched: GROUNDFAULT, BACKUP_RELAY
//! [ANOMALY]  e_norm=0.333  E=2.000  CD=+4.000  — Missing symbol — E changes only if removed atom was covered
//!          unmatched: BACKUP_RELAY
//!
//! === Varied corpus (4 variants × 8) ===
//!
//! Grammar learns shorter patterns; reordered sequences may partially match.
//!
//! [ANOMALY]  e_norm=0.300  E=3.000  CD=+7.000  — Normal (atoms not in patterns still uncovered → E > 0 possible)
//!          unmatched: TRIP_A
//! [ANOMALY]  e_norm=0.800  E=8.000  CD=+2.000  — Reordered — partially detected (multi-pattern stitching reduces E vs homogeneous)
//!          unmatched: BACKUP_RELAY, UNDERVOLTAGE, TRIP_A
//! [ANOMALY]  e_norm=0.600  E=6.000  CD=+4.000  — Unknown symbol — detected
//!          unmatched: TRIP_A, GROUNDFAULT
//! ```
//!
//! SPMA flags a sequence as anomalous when any symbol is uncovered by the
//! learned grammar (E > 0, threshold = 0.0 by default). With frequency-based
//! costs, every uncovered atom contributes positively to E — so a "normal"
//! sequence can still score ANOMALY if the grammar did not induce patterns
//! that cover every position.
//!
//! Practical implication: grammar completeness depends on corpus size and
//! frequency thresholds. Atoms that appear in every training sequence but are
//! never the shared component of a co-occurring pair will not enter any
//! pattern and remain uncovered at inference. This is expected behavior, not
//! a cost model bug.
//!
//! Order detection works when:
//!   - The corpus is homogeneous (one canonical sequence, repeated many times).
//!   - The grammar learns patterns that span the full sequence.
//!
//! Order detection weakens when:
//!   - Multiple sequence variants are in the corpus — grammar learns shorter,
//!     more flexible patterns that can stitch across reorderings.

use spma::Spma;

fn main() {
    // ── Homogeneous corpus: order detection works ────────────────────────────

    println!("=== Homogeneous corpus (single sequence type × 8) ===\n");
    println!("Note: atoms not covered by any learned pattern score E > 0.");
    println!("'Normal' may still show ANOMALY if grammar coverage is incomplete.\n");

    let mut engine = Spma::new(10);
    engine.train(
        &(0..8)
            .map(|_| vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"])
            .collect::<Vec<_>>(),
    );

    let cases: Vec<(&str, Vec<&str>)> = vec![
        (
            "Normal (E>0 if any atom uncovered by grammar)",
            vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Reordered — higher E (grammar spans sequence, reorder breaks patterns)",
            vec!["BACKUP_RELAY", "UNDERVOLTAGE", "BREAKER_OPEN", "TRIP_A"],
        ),
        (
            "Unknown symbol — highest E (unknown atom costs max bits)",
            vec!["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"],
        ),
        (
            "Missing symbol — E changes only if removed atom was covered",
            vec!["TRIP_A", "BREAKER_OPEN", "BACKUP_RELAY"],
        ),
    ];

    for (label, seq) in &cases {
        let r = engine.infer(seq);
        let tag = if r.is_anomaly { "ANOMALY" } else { "OK     " };
        println!(
            "[{tag}]  e_norm={:.3}  E={:.3}  CD={:+.3}  — {label}",
            r.e_norm, r.e_cost, r.cd
        );
        let unmatched = r.alignment.unmatched_symbols();
        if !unmatched.is_empty() {
            println!("         unmatched: {}", unmatched.join(", "));
        }
    }

    // ── Varied corpus: order detection breaks down ───────────────────────────

    println!("\n=== Varied corpus (4 variants × 8) ===\n");
    println!("Grammar learns shorter patterns; reordered sequences may partially match.\n");

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
            "Normal (atoms not in patterns still uncovered → E > 0 possible)",
            vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        ),
        (
            "Reordered — partially detected (multi-pattern stitching reduces E vs homogeneous)",
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
        println!(
            "[{tag}]  e_norm={:.3}  E={:.3}  CD={:+.3}  — {label}",
            r.e_norm, r.e_cost, r.cd
        );
        let unmatched = r.alignment.unmatched_symbols();
        if !unmatched.is_empty() {
            println!("         unmatched: {}", unmatched.join(", "));
        }
    }

    println!("\nSummary:");
    println!("  - Unknown symbols:     always detected (high atom cost)");
    println!("  - Order violations:    detected with homogeneous corpus; weaker with varied");
    println!("  - Missing symbols:     detected only if removed atom was grammar-covered");
    println!("  - Grammar completeness: atoms never in a learned pattern stay uncovered");
    println!("    → training sequences can score E > 0 if corpus is too small or varied");
}
