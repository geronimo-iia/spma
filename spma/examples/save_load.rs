//! Demonstrates saving a trained model to disk and reloading it.
//!
//! The loaded model produces identical inference results to the original.

use spma::Spma;
use std::io::{BufReader, BufWriter};

fn main() -> std::io::Result<()> {
    let corpus: Vec<Vec<&str>> = vec![
        vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_A", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_A", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "UNDERVOLTAGE", "BACKUP_RELAY"],
        vec!["TRIP_B", "BREAKER_OPEN", "OVERCURRENT", "BACKUP_RELAY"],
    ];

    let mut model = Spma::new(10);
    model.train(&corpus);

    let path = "/tmp/spma_example_model.json";

    // Save
    model.save(BufWriter::new(std::fs::File::create(path)?))?;
    println!("Saved model to {path}");

    // Load
    let loaded = Spma::load(BufReader::new(std::fs::File::open(path)?))?;
    println!("Loaded model from {path}");

    // Verify identical results
    let seq = &["TRIP_A", "BREAKER_OPEN", "GROUNDFAULT", "BACKUP_RELAY"];
    let r_orig = model.infer(seq);
    let r_load = loaded.infer(seq);

    println!(
        "\nOriginal: e_norm={:.3}  anomaly={}",
        r_orig.e_norm, r_orig.is_anomaly
    );
    println!(
        "Loaded:   e_norm={:.3}  anomaly={}",
        r_load.e_norm, r_load.is_anomaly
    );

    assert_eq!(r_orig.is_anomaly, r_load.is_anomaly);
    assert!((r_orig.e_norm - r_load.e_norm).abs() < 1e-10);
    println!("\nResults identical ✓");

    std::fs::remove_file(path)?;
    Ok(())
}
