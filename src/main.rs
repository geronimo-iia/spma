use anyhow::Result;
use clap::{Arg, ArgAction, Command};
use spma::Spma;
use std::fs;

fn main() -> Result<()> {
    let matches = Command::new("spma")
        .version("0.1.0")
        .about("SP Multiple Alignment — symbolic anomaly detection via T=G+E scoring")
        .subcommand(
            Command::new("train")
                .about("Learn grammar from input file, save to grammar path")
                .arg(
                    Arg::new("input")
                        .required(true)
                        .index(1)
                        .help("Input file (one sequence per line)"),
                )
                .arg(
                    Arg::new("grammar")
                        .long("grammar")
                        .value_name("PATH")
                        .default_value("spma_grammar.bin")
                        .help("Path to save the learned grammar"),
                )
                .arg(
                    Arg::new("verbose")
                        .long("verbose")
                        .short('v')
                        .action(ArgAction::SetTrue)
                        .help("Print alignment tables during training"),
                ),
        )
        .subcommand(
            Command::new("infer")
                .about("Load grammar and align each line; exit 1 if any anomaly detected (E > 0)")
                .arg(
                    Arg::new("input")
                        .required(true)
                        .index(1)
                        .help("Input file (one sequence per line)"),
                )
                .arg(
                    Arg::new("grammar")
                        .long("grammar")
                        .value_name("PATH")
                        .default_value("spma_grammar.bin")
                        .help("Path to saved grammar"),
                )
                .arg(
                    Arg::new("verbose")
                        .long("verbose")
                        .short('v')
                        .action(ArgAction::SetTrue)
                        .help("Print full alignment tables"),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("train", sub)) => {
            let input_file = sub.get_one::<String>("input").unwrap();
            let grammar_path = sub.get_one::<String>("grammar").unwrap();
            let verbose = sub.get_flag("verbose");

            let content = fs::read_to_string(input_file)?;
            let sequences: Vec<Vec<&str>> = content
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
                .map(|l| l.split_whitespace().collect())
                .collect();

            if sequences.is_empty() {
                eprintln!("No sequences in {}", input_file);
                std::process::exit(1);
            }

            println!(
                "Training on {} sequences from {}",
                sequences.len(),
                input_file
            );

            let mut engine = Spma::new();
            engine.train(&sequences)?;
            engine.save(grammar_path)?;

            println!("Grammar saved to {}", grammar_path);

            if verbose {
                // Re-run infer with full output for each sequence
                let engine2 = Spma::load(grammar_path)?;
                for seq in &sequences {
                    let result = engine2.infer(seq)?;
                    println!("{}", result.alignment);
                }
            }
        }

        Some(("infer", sub)) => {
            let input_file = sub.get_one::<String>("input").unwrap();
            let grammar_path = sub.get_one::<String>("grammar").unwrap();
            let verbose = sub.get_flag("verbose");

            let content = fs::read_to_string(input_file)?;
            let sequences: Vec<Vec<&str>> = content
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
                .map(|l| l.split_whitespace().collect())
                .collect();

            let engine = Spma::load(grammar_path)?;
            let mut any_anomaly = false;

            for seq in &sequences {
                let result = engine.infer(seq)?;
                if result.is_anomaly {
                    any_anomaly = true;
                    print!("ANOMALY  ");
                } else {
                    print!("OK       ");
                }
                println!(
                    "E={:.2}  CD={:+.2}  seq: {}",
                    result.e_cost,
                    result.cd,
                    seq.join(" ")
                );
                if verbose {
                    println!("{}", result.alignment);
                }
            }

            if any_anomaly {
                std::process::exit(1);
            }
        }

        _ => {
            eprintln!("Usage: spma <train|infer> <input_file> [--grammar <path>]");
            eprintln!("       spma help");
            std::process::exit(1);
        }
    }

    Ok(())
}
