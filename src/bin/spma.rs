use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use spma::engine::Spma;

#[derive(Parser)]
#[command(name = "spma", about = "Sparse Pattern Matching Anomaly detector")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a model on a corpus file and save it
    Train {
        /// Path to corpus file (one sequence per line, tokens space-separated)
        #[arg(short, long)]
        corpus: String,

        /// Output model path
        #[arg(short, long)]
        output: String,

        /// Beam width (default: 10)
        #[arg(short, long, default_value_t = 10)]
        beam: usize,

        /// Max induced gap size (default: 3)
        #[arg(short, long, default_value_t = 3)]
        max_gap: usize,

        /// Anomaly threshold (e_norm cutoff, default: auto from training distribution)
        #[arg(short, long)]
        threshold: Option<f64>,
    },

    /// Run inference on sequences from stdin or a file
    Infer {
        /// Path to saved model
        #[arg(short, long)]
        model: String,

        /// Input file (one sequence per line, tokens space-separated); defaults to stdin
        #[arg(short, long)]
        input: Option<String>,

        /// Emit JSON output instead of plain text
        #[arg(long)]
        json: bool,
    },
}

fn read_corpus(path: &str) -> Result<Vec<Vec<String>>> {
    let f = File::open(path).with_context(|| format!("open corpus: {path}"))?;
    let reader = BufReader::new(f);
    let mut corpus = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim().to_owned();
        if trimmed.is_empty() {
            continue;
        }
        let tokens: Vec<String> = trimmed.split_whitespace().map(str::to_owned).collect();
        corpus.push(tokens);
    }
    Ok(corpus)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Train {
            corpus,
            output,
            beam,
            max_gap,
            threshold,
        } => {
            let raw = read_corpus(&corpus)?;
            let corpus_refs: Vec<Vec<&str>> = raw
                .iter()
                .map(|seq| seq.iter().map(String::as_str).collect())
                .collect();

            let mut spma = Spma::new(beam);
            spma.set_max_induced_gap(max_gap);
            spma.train(&corpus_refs);

            if let Some(t) = threshold {
                spma.set_anomaly_threshold(t);
            }

            let f = File::create(&output).with_context(|| format!("create model: {output}"))?;
            spma.save(BufWriter::new(f))
                .with_context(|| format!("save model: {output}"))?;

            let dist = spma.e_distribution();
            eprintln!(
                "trained: {} sequences, {} grammar levels, threshold={:.4}",
                raw.len(),
                spma.grammar.levels.len(),
                dist.threshold
            );
        }

        Command::Infer { model, input, json } => {
            let f = File::open(&model).with_context(|| format!("open model: {model}"))?;
            let spma =
                Spma::load(BufReader::new(f)).with_context(|| format!("load model: {model}"))?;

            let reader: Box<dyn io::Read> = match input {
                Some(ref path) => {
                    Box::new(File::open(path).with_context(|| format!("open input: {path}"))?)
                }
                None => Box::new(io::stdin()),
            };

            let buf = BufReader::new(reader);
            let mut any_anomaly = false;
            for line in buf.lines() {
                let line = line?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                let result = spma.infer(&tokens);

                if result.is_anomaly {
                    any_anomaly = true;
                }

                if json {
                    println!(
                        "{{\"seq\":{:?},\"e_cost\":{:.6},\"e_norm\":{:.6},\"cd\":{:.6},\"anomaly_percentile\":{:.6},\"is_anomaly\":{}}}",
                        tokens,
                        result.e_cost,
                        result.e_norm,
                        result.cd,
                        result.anomaly_percentile,
                        result.is_anomaly,
                    );
                } else {
                    println!(
                        "{}\te_norm={:.4}\tpct={:.4}\t{}",
                        trimmed,
                        result.e_norm,
                        result.anomaly_percentile,
                        if result.is_anomaly { "ANOMALY" } else { "ok" },
                    );
                }
            }

            if any_anomaly {
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
