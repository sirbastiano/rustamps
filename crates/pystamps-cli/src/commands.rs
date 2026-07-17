use std::fs;
use std::path::PathBuf;

use clap::{Args, Subcommand};
use serde_json::json;

#[derive(Debug, Args)]
pub struct RunArgs {
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long, default_value_t = 1)]
    start_step: u8,
    #[arg(long, default_value_t = 8)]
    end_step: u8,
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    cpu_workers: Option<usize>,
}

#[derive(Debug, Args)]
pub struct PrepArgs {
    #[command(subcommand)]
    command: PrepCommand,
}

#[derive(Debug, Subcommand)]
enum PrepCommand {
    Snap(SnapPrepArgs),
}

#[derive(Debug, Args)]
struct SnapPrepArgs {
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    master_date: Option<String>,
    #[arg(long, default_value_t = 0.4)]
    amp_dispersion: f64,
    #[arg(long, default_value_t = 1)]
    range_patches: usize,
    #[arg(long, default_value_t = 1)]
    azimuth_patches: usize,
    #[arg(long, default_value_t = 50)]
    range_overlap: usize,
    #[arg(long, default_value_t = 50)]
    azimuth_overlap: usize,
    #[arg(long)]
    force: bool,
}

pub fn run_pipeline_command(
    args: RunArgs,
    mut config: pystamps_pipeline::RunConfig,
) -> Result<(), String> {
    if let Some(workers) = args.cpu_workers {
        config.runtime.cpu_workers = workers;
    }
    let context = pystamps_pipeline::PipelineContext {
        dataset_root: args.dataset,
        config,
        start_step: args.start_step,
        end_step: args.end_step,
        dry_run: args.dry_run,
    };
    let report = pystamps_pipeline::run_pipeline(&context, &pystamps_pipeline::NativeExecutor)
        .map_err(|error| error.to_string())?;
    println!("{}", serde_json::to_string_pretty(&report.results).unwrap());
    if report.ok() {
        Ok(())
    } else {
        Err("pipeline reported failures".to_owned())
    }
}

pub fn status(dataset: PathBuf) -> Result<(), String> {
    let status = pystamps_pipeline::collect_status(dataset).map_err(|error| error.to_string())?;
    println!("{}", serde_json::to_string_pretty(&status).unwrap());
    Ok(())
}

pub fn verify(
    run: PathBuf,
    golden: PathBuf,
    config: pystamps_pipeline::RunConfig,
    through_stage: Option<u8>,
    final_products_only: bool,
) -> Result<(), String> {
    let report = pystamps_verify::verify_paths_with_scope(
        &run,
        &golden,
        &config.tolerance,
        through_stage,
        final_products_only,
    )
    .map_err(|error| error.to_string())?;
    let tolerated = report
        .comparisons
        .iter()
        .filter(|item| item.ok && !item.outliers.is_empty())
        .collect::<Vec<_>>();
    let payload = json!({
        "ok": report.ok(),
        "profile": config.tolerance.profile,
        "through_stage": through_stage,
        "final_products_only": final_products_only,
        "checked": report.comparisons.len(),
        "failed": report.comparisons.iter().filter(|item| !item.ok).collect::<Vec<_>>(),
        "tolerated": tolerated,
    });
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    report
        .ok()
        .then_some(())
        .ok_or_else(|| "verification failed".to_owned())
}

pub fn prep(args: PrepArgs) -> Result<(), String> {
    match args.command {
        PrepCommand::Snap(args) => {
            let summary = pystamps_io::prepare_snap(
                &args.dataset,
                pystamps_io::SnapPrepOptions {
                    master_date: args.master_date.as_deref(),
                    amp_dispersion: args.amp_dispersion,
                    range_patches: args.range_patches,
                    azimuth_patches: args.azimuth_patches,
                    range_overlap: args.range_overlap,
                    azimuth_overlap: args.azimuth_overlap,
                    force: args.force,
                },
            )
            .map_err(|error| error.to_string())?;
            println!("{}", serde_json::to_string_pretty(&summary).unwrap());
            Ok(())
        }
    }
}

pub fn list_legacy(root: PathBuf) -> Result<(), String> {
    let mut scripts = Vec::new();
    collect_scripts(&root, &mut scripts).map_err(|error| error.to_string())?;
    scripts.sort();
    println!("{}", serde_json::to_string_pretty(&scripts).unwrap());
    Ok(())
}

fn collect_scripts(root: &std::path::Path, scripts: &mut Vec<String>) -> std::io::Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_scripts(&path, scripts)?;
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("m" | "csh" | "sh")
        ) {
            scripts.push(path.display().to_string());
        }
    }
    Ok(())
}

pub fn describe_inputs(
    stage: String,
    dataset: Option<PathBuf>,
    patch: String,
) -> Result<(), String> {
    let stages = parse_stages(&stage)?;
    let payload = json!({
        "stages": stages.iter().map(|stage| json!({
            "stage": stage,
            "scope": if *stage <= 5 { "patch" } else { "merged" },
        })).collect::<Vec<_>>(),
        "stage1_dataset_check": dataset.map(|root| json!({
            "dataset": root,
            "patch": patch,
            "implemented_by": "native",
        })),
    });
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    Ok(())
}

fn parse_stages(value: &str) -> Result<Vec<u8>, String> {
    if value.eq_ignore_ascii_case("all") {
        return Ok((1..=8).collect());
    }
    value
        .split(',')
        .map(|part| {
            let stage: u8 = part
                .trim()
                .parse()
                .map_err(|_| format!("invalid stage {part}"))?;
            (1..=8)
                .contains(&stage)
                .then_some(stage)
                .ok_or_else(|| format!("stage {stage} is outside 1..8"))
        })
        .collect()
}

pub fn describe_backends() -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "providers": {
                "native": {"description": "Standalone Rust backend", "available": true, "aliases": ["auto"]}
            },
            "runtime_external_dependencies": [],
        }))
        .unwrap()
    );
    Ok(())
}
