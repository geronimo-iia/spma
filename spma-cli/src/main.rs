#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
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

    /// Print grammar summary (human-readable or JSON)
    Grammar {
        /// Path to saved model
        #[arg(short, long)]
        model: String,

        /// Emit JSON output instead of human-readable text
        #[arg(long)]
        json: bool,

        /// Restrict output to a single grammar level (0-indexed)
        #[arg(short, long)]
        level: Option<usize>,
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

        /// Override anomaly threshold from model (e_norm cutoff).
        /// If omitted, uses value stored in the model.
        #[arg(short, long)]
        threshold: Option<f64>,

        /// Set per-level threshold as "level:value" pairs, e.g. --level-threshold 1:0.3
        /// Can be specified multiple times. Falls back to global --threshold if not set.
        #[arg(long, value_parser = parse_level_threshold)]
        level_threshold: Vec<(usize, f64)>,
    },

    /// Extend an existing model with a new batch of sequences without cold start
    Retrain {
        /// Path to saved model (modified in place or written to --output)
        #[arg(short, long)]
        model: String,

        /// New corpus to train on (one sequence per line, tokens space-separated)
        #[arg(short, long)]
        corpus: String,

        /// Output path; if omitted, overwrites --model
        #[arg(short, long)]
        output: Option<String>,

        /// Override anomaly threshold after retraining
        #[arg(short, long)]
        threshold: Option<f64>,
    },

    /// Reload a model, replay corpus to refit e_distribution, save updated model
    Recalibrate {
        /// Path to saved model (modified in place or written to --output)
        #[arg(short, long)]
        model: String,

        /// Training corpus to replay (one sequence per line, tokens space-separated)
        #[arg(short, long)]
        corpus: String,

        /// Output path; if omitted, overwrites --model
        #[arg(short, long)]
        output: Option<String>,

        /// Override anomaly threshold after recalibration
        #[arg(short, long)]
        threshold: Option<f64>,
    },
}

fn parse_level_threshold(s: &str) -> Result<(usize, f64), String> {
    let (l, v) = s
        .split_once(':')
        .ok_or_else(|| format!("expected level:value, got {s:?}"))?;
    let level = l.parse::<usize>().map_err(|e| e.to_string())?;
    let value = v.parse::<f64>().map_err(|e| e.to_string())?;
    Ok((level, value))
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

fn quantile_of(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((q * sorted.len() as f64) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn render_symbol(
    sym: &spma::model::SymbolRef,
    interner: &spma::Interner,
    level_idx: usize,
) -> String {
    use spma::model::SymbolRef;
    match sym {
        SymbolRef::Atom(id) => interner.name(*id).to_owned(),
        SymbolRef::Pattern(id) => {
            let _ = level_idx;
            format!("P{id}")
        }
    }
}

fn render_pattern_human(
    pat: &spma::model::Pattern,
    interner: &spma::Interner,
    level_idx: usize,
    total_freq: u64,
) -> String {
    let pct = if total_freq > 0 {
        pat.frequency as f64 / total_freq as f64 * 100.0
    } else {
        0.0
    };
    let mut parts: Vec<String> = Vec::new();
    for (i, sym) in pat.symbols.iter().enumerate() {
        if i > 0 {
            if let Some(gap) = pat.gap_between(i - 1) {
                if gap.min == 0 && gap.max == 0 {
                    parts.push("→".to_owned());
                } else {
                    parts.push(format!("~[{},{}]→", gap.min, gap.max));
                }
            } else {
                parts.push("→".to_owned());
            }
        }
        parts.push(render_symbol(sym, interner, level_idx));
    }
    format!(
        "  [idx={:<3} freq={:>7}  {:>5.1}%] {}",
        pat.id,
        pat.frequency,
        pct,
        parts.join(" ")
    )
}

fn print_grammar_human(
    out: &mut impl Write,
    spma: &Spma,
    level_filter: Option<usize>,
    model_path: &str,
) -> Result<()> {
    let grammar = &spma.grammar();
    let interner = &grammar.interner;
    let dist = &grammar.e_distribution;

    writeln!(out, "Model: {model_path}")?;
    writeln!(
        out,
        "Beam: {}  MaxGap: {}",
        spma.beam_k(),
        spma.max_induced_gap()
    )?;

    let n_atoms = interner.len();
    let atom_names: Vec<&str> = (0..n_atoms as u32).map(|i| interner.name(i)).collect();
    writeln!(out, "Atoms ({}): {}", n_atoms, atom_names.join(" "))?;
    writeln!(out)?;

    // Atom costs
    writeln!(out, "Atom costs:")?;
    for (i, &cost) in spma.atom_costs().iter().enumerate() {
        let name = interner.name(i as u32);
        let bar_len = (cost * 10.0).round() as usize;
        let bar = "█".repeat(bar_len);
        writeln!(out, "  {:<5} {:.3}  {}", name, cost, bar)?;
    }
    writeln!(out)?;

    // Uncovered atoms
    let mut covered = vec![false; n_atoms];
    for level in &grammar.levels {
        for pat in &level.patterns {
            for sym in &pat.symbols {
                if let spma::model::SymbolRef::Atom(id) = sym {
                    if (*id as usize) < covered.len() {
                        covered[*id as usize] = true;
                    }
                }
            }
        }
    }
    let uncovered: Vec<&str> = (0..n_atoms)
        .filter(|&i| !covered[i])
        .map(|i| interner.name(i as u32))
        .collect();
    if !uncovered.is_empty() {
        writeln!(
            out,
            "Uncovered atoms (always contribute to e_cost): {}",
            uncovered.join(" ")
        )?;
    }
    writeln!(out)?;

    let total_patterns: usize = grammar.levels.iter().map(|l| l.patterns.len()).sum();
    writeln!(
        out,
        "Grammar: {} levels, {} patterns",
        grammar.levels.len(),
        total_patterns
    )?;
    writeln!(out)?;

    let levels_to_print: Vec<usize> = match level_filter {
        Some(l) => vec![l],
        None => (0..grammar.levels.len()).collect(),
    };

    for li in &levels_to_print {
        let li = *li;
        let Some(level) = grammar.levels.get(li) else {
            writeln!(out, "Level {li}: not found")?;
            continue;
        };
        let gap_count = level.patterns.iter().filter(|p| !p.is_contiguous()).count();
        let total_freq: u64 = level.patterns.iter().map(|p| p.frequency as u64).sum();
        writeln!(
            out,
            "Level {li}: {} patterns, {} gap patterns, total_freq={total_freq}",
            level.patterns.len(),
            gap_count,
        )?;
        for pat in &level.patterns {
            writeln!(
                out,
                "{}",
                render_pattern_human(pat, interner, li, total_freq)
            )?;
        }
        writeln!(out)?;
    }

    // E_norm distribution table
    writeln!(out, "E_norm distribution per level (training):")?;
    writeln!(
        out,
        "  {:>5}  {:>6}  {:>7}  {:>7}  {:>7}  {:>7}",
        "level", "n", "p50", "p90", "p99", "max"
    )?;
    for (li, sorted) in dist.level_sorted_e_norms.iter().enumerate() {
        if level_filter.is_some() && level_filter != Some(li) {
            continue;
        }
        let n = sorted.len();
        let p50 = quantile_of(sorted, 0.50);
        let p90 = quantile_of(sorted, 0.90);
        let p99 = quantile_of(sorted, 0.99);
        let max = sorted.last().copied().unwrap_or(0.0);
        writeln!(
            out,
            "  {:>5}  {:>6}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}",
            li, n, p50, p90, p99, max
        )?;
    }

    Ok(())
}

fn print_grammar_json(
    out: &mut impl Write,
    spma: &Spma,
    level_filter: Option<usize>,
    model_path: &str,
) -> Result<()> {
    use serde_json::{json, Value};

    let grammar = &spma.grammar();
    let interner = &grammar.interner;
    let dist = &grammar.e_distribution;
    let n_atoms = interner.len();

    // atoms array
    let atoms: Vec<Value> = (0..n_atoms)
        .map(|i| {
            let name = interner.name(i as u32);
            let cost = spma.atom_costs().get(i).copied().unwrap_or(0.0);
            json!({"id": i, "name": name, "cost": cost})
        })
        .collect();

    // uncovered atoms
    let mut covered = vec![false; n_atoms];
    for level in &grammar.levels {
        for pat in &level.patterns {
            for sym in &pat.symbols {
                if let spma::model::SymbolRef::Atom(id) = sym {
                    if (*id as usize) < covered.len() {
                        covered[*id as usize] = true;
                    }
                }
            }
        }
    }
    let uncovered_atoms: Vec<&str> = (0..n_atoms)
        .filter(|&i| !covered[i])
        .map(|i| interner.name(i as u32))
        .collect();

    // levels array
    let levels_to_emit: Vec<usize> = match level_filter {
        Some(l) => vec![l],
        None => (0..grammar.levels.len()).collect(),
    };

    let levels: Vec<Value> = levels_to_emit
        .iter()
        .filter_map(|&li| {
            let level = grammar.levels.get(li)?;
            let gap_count = level.patterns.iter().filter(|p| !p.is_contiguous()).count();
            let total_freq: u64 = level.patterns.iter().map(|p| p.frequency as u64).sum();

            let patterns: Vec<Value> = level
                .patterns
                .iter()
                .map(|pat| {
                    let freq_pct = if total_freq > 0 {
                        (pat.frequency as f64 / total_freq as f64 * 1000.0).round() / 10.0
                    } else {
                        0.0
                    };

                    let symbols: Vec<Value> = pat
                        .symbols
                        .iter()
                        .map(|sym| match sym {
                            spma::model::SymbolRef::Atom(id) => {
                                let name = interner.name(*id);
                                json!({"kind": "atom", "id": *id, "name": name})
                            }
                            spma::model::SymbolRef::Pattern(id) => {
                                json!({"kind": "pattern", "id": *id})
                            }
                        })
                        .collect();

                    let gaps: Vec<Value> = if pat.gaps.is_empty() {
                        vec![]
                    } else {
                        pat.gaps
                            .iter()
                            .map(|g| json!({"min": g.min, "max": g.max}))
                            .collect()
                    };

                    // rendered string
                    let mut rendered_parts: Vec<String> = Vec::new();
                    for (i, sym) in pat.symbols.iter().enumerate() {
                        if i > 0 {
                            if let Some(gap) = pat.gap_between(i - 1) {
                                if gap.min == 0 && gap.max == 0 {
                                    rendered_parts.push("→".to_owned());
                                } else {
                                    rendered_parts.push(format!("~[{},{}]→", gap.min, gap.max));
                                }
                            } else {
                                rendered_parts.push("→".to_owned());
                            }
                        }
                        match sym {
                            spma::model::SymbolRef::Atom(id) => {
                                rendered_parts.push(interner.name(*id).to_owned());
                            }
                            spma::model::SymbolRef::Pattern(id) => {
                                rendered_parts.push(format!("P{id}"));
                            }
                        }
                    }
                    let rendered = rendered_parts.join(" ");

                    json!({
                        "idx": pat.id,
                        "frequency": pat.frequency,
                        "frequency_pct": freq_pct,
                        "symbols": symbols,
                        "gaps": gaps,
                        "rendered": rendered,
                    })
                })
                .collect();

            let e_norm_val: Option<Value> = dist.level_sorted_e_norms.get(li).map(|sorted| {
                let n = sorted.len();
                let p50 = quantile_of(sorted, 0.50);
                let p90 = quantile_of(sorted, 0.90);
                let p99 = quantile_of(sorted, 0.99);
                let max = sorted.last().copied().unwrap_or(0.0);
                json!({"n": n, "p50": p50, "p90": p90, "p99": p99, "max": max})
            });

            let mut obj = json!({
                "level": li,
                "pattern_count": level.patterns.len(),
                "gap_pattern_count": gap_count,
                "total_frequency": total_freq,
                "patterns": patterns,
            });
            if let Some(e) = e_norm_val {
                obj["e_norm"] = e;
            }
            Some(obj)
        })
        .collect();

    let atom_costs: Vec<f64> = spma.atom_costs().to_vec();

    let output = json!({
        "model_path": model_path,
        "beam_k": spma.beam_k(),
        "max_induced_gap": spma.max_induced_gap(),
        "atoms": atoms,
        "uncovered_atoms": uncovered_atoms,
        "levels": levels,
        "threshold": dist.threshold,
        "atom_costs": atom_costs,
    });

    serde_json::to_writer_pretty(&mut *out, &output)?;
    writeln!(out)?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Grammar { model, json, level } => {
            let f = File::open(&model).with_context(|| format!("open model: {model}"))?;
            let spma =
                Spma::load(BufReader::new(f)).with_context(|| format!("load model: {model}"))?;
            let stdout = io::stdout();
            let mut out = BufWriter::new(stdout.lock());
            if json {
                print_grammar_json(&mut out, &spma, level, &model)?;
            } else {
                print_grammar_human(&mut out, &spma, level, &model)?;
            }
            out.flush()?;
        }

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
                spma.grammar().levels.len(),
                dist.threshold
            );
        }

        Command::Infer {
            model,
            input,
            json,
            threshold,
            level_threshold,
        } => {
            let f = File::open(&model).with_context(|| format!("open model: {model}"))?;
            let mut spma =
                Spma::load(BufReader::new(f)).with_context(|| format!("load model: {model}"))?;

            if let Some(t) = threshold {
                spma.set_anomaly_threshold(t);
            }

            for (level, t) in &level_threshold {
                spma.set_level_threshold(*level, *t);
            }

            let reader: Box<dyn io::Read> = match input {
                Some(ref path) => {
                    Box::new(File::open(path).with_context(|| format!("open input: {path}"))?)
                }
                None => Box::new(io::stdin()),
            };

            let buf = BufReader::new(reader);
            let lines: Vec<String> = buf
                .lines()
                .map(|l| l.map(|s| s.trim().to_owned()))
                .collect::<io::Result<Vec<_>>>()?
                .into_iter()
                .filter(|l| !l.is_empty())
                .collect();

            let results: Vec<(Vec<String>, spma::engine::InferResult)> = lines
                .par_iter()
                .map(|line| {
                    let tokens: Vec<&str> = line.split_whitespace().collect();
                    let result = spma.infer(&tokens);
                    (tokens.iter().map(|s| s.to_string()).collect(), result)
                })
                .collect();

            let stdout = io::stdout();
            let mut out = BufWriter::new(stdout.lock());
            let mut any_anomaly = false;

            for (tokens, result) in &results {
                if result.is_anomaly {
                    any_anomaly = true;
                }
                if json {
                    writeln!(
                        out,
                        "{{\"seq\":{:?},\"e_cost\":{:.6},\"e_norm\":{:.6},\"cd\":{:.6},\"anomaly_percentile\":{:.6},\"is_anomaly\":{}}}",
                        tokens,
                        result.e_cost,
                        result.e_norm,
                        result.cd,
                        result.anomaly_percentile,
                        result.is_anomaly,
                    )?;
                } else {
                    writeln!(
                        out,
                        "{}\te_norm={:.4}\tpct={:.4}\t{}",
                        tokens.join(" "),
                        result.e_norm,
                        result.anomaly_percentile,
                        if result.is_anomaly { "ANOMALY" } else { "ok" },
                    )?;
                }
            }
            out.flush()?;

            if any_anomaly {
                std::process::exit(1);
            }
        }

        Command::Retrain {
            model,
            corpus,
            output,
            threshold,
        } => {
            let f = File::open(&model).with_context(|| format!("open model: {model}"))?;
            let mut spma =
                Spma::load(BufReader::new(f)).with_context(|| format!("load model: {model}"))?;

            let raw = read_corpus(&corpus)?;
            if raw.is_empty() {
                anyhow::bail!("corpus is empty: {corpus}");
            }
            let corpus_refs: Vec<Vec<&str>> = raw
                .iter()
                .map(|seq| seq.iter().map(String::as_str).collect())
                .collect();

            let levels_before = spma.grammar().levels.len();
            spma.retrain(&corpus_refs);

            if let Some(t) = threshold {
                spma.set_anomaly_threshold(t);
            }

            let out_path = output.as_deref().unwrap_or(&model);
            let out_dir = std::path::Path::new(out_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            let tmp_path = out_dir.join(format!(".spma_retrain_tmp_{}", std::process::id()));
            let f = File::create(&tmp_path)
                .with_context(|| format!("create tmp output: {}", tmp_path.display()))?;
            spma.save(BufWriter::new(f))
                .with_context(|| format!("save model to tmp: {}", tmp_path.display()))?;
            std::fs::rename(&tmp_path, out_path)
                .with_context(|| format!("rename {} -> {out_path}", tmp_path.display()))?;

            eprintln!(
                "retrained: {} sequences, levels {} -> {}, threshold={:.4}",
                raw.len(),
                levels_before,
                spma.grammar().levels.len(),
                spma.e_distribution().threshold,
            );
        }

        Command::Recalibrate {
            model,
            corpus,
            output,
            threshold,
        } => {
            let f = File::open(&model).with_context(|| format!("open model: {model}"))?;
            let mut spma =
                Spma::load(BufReader::new(f)).with_context(|| format!("load model: {model}"))?;

            let raw = read_corpus(&corpus)?;
            if raw.is_empty() {
                anyhow::bail!("corpus is empty: {corpus}");
            }
            let corpus_refs: Vec<Vec<&str>> = raw
                .iter()
                .map(|seq| seq.iter().map(String::as_str).collect())
                .collect();

            spma.recalibrate(&corpus_refs);

            if let Some(t) = threshold {
                spma.set_anomaly_threshold(t);
            }

            let out_path = output.as_deref().unwrap_or(&model);
            let out_dir = std::path::Path::new(out_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
            let tmp_path = out_dir.join(format!(".spma_recal_tmp_{}", std::process::id()));
            let f = File::create(&tmp_path)
                .with_context(|| format!("create tmp output: {}", tmp_path.display()))?;
            spma.save(BufWriter::new(f))
                .with_context(|| format!("save model to tmp: {}", tmp_path.display()))?;
            std::fs::rename(&tmp_path, out_path)
                .with_context(|| format!("rename {} -> {out_path}", tmp_path.display()))?;

            let dist = spma.e_distribution();
            eprintln!(
                "recalibrated: {} sequences, threshold={:.4}",
                raw.len(),
                dist.threshold
            );
        }
    }

    Ok(())
}
