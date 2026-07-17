mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use commands::{PrepArgs, RunArgs};

#[derive(Debug, Parser)]
#[command(
    name = "pystamps",
    version,
    about = "Native StaMPS-compatible processing runtime"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Run(RunArgs),
    Status {
        #[arg(long)]
        dataset: PathBuf,
    },
    Verify {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        golden: PathBuf,
        #[arg(long, value_enum)]
        profile: Option<VerifyProfile>,
        #[arg(
            long,
            value_parser = clap::value_parser!(u8).range(1..=8),
            help = "Compare only production artifacts through stage 1..8"
        )]
        through_stage: Option<u8>,
        #[arg(
            long,
            help = "Compare final stage products while excluding grid/cache intermediates"
        )]
        final_products_only: bool,
    },
    Prep(PrepArgs),
    ListLegacy {
        #[arg(long, env = "STAMPS_ROOT")]
        stamps_root: PathBuf,
    },
    DescribeInputs {
        #[arg(long, default_value = "all")]
        stage: String,
        #[arg(long)]
        dataset: Option<PathBuf>,
        #[arg(long, default_value = "PATCH_1")]
        patch: String,
    },
    DescribeBackends,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum VerifyProfile {
    Strict,
    Scientific,
}

impl From<VerifyProfile> for pystamps_pipeline::config::VerificationProfile {
    fn from(value: VerifyProfile) -> Self {
        match value {
            VerifyProfile::Strict => Self::Strict,
            VerifyProfile::Scientific => Self::Scientific,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let profile = match &cli.command {
        Command::Verify { profile, .. } => profile.map(Into::into),
        _ => None,
    };
    let config = pystamps_pipeline::load_config_with_profile(cli.config.as_deref(), profile)
        .map_err(|error| format!("Config error: {error}"))?;
    match cli.command {
        Command::Run(args) => commands::run_pipeline_command(args, config),
        Command::Status { dataset } => commands::status(dataset),
        Command::Verify {
            run,
            golden,
            through_stage,
            final_products_only,
            ..
        } => commands::verify(run, golden, config, through_stage, final_products_only),
        Command::Prep(args) => commands::prep(args),
        Command::ListLegacy { stamps_root } => commands::list_legacy(stamps_root),
        Command::DescribeInputs {
            stage,
            dataset,
            patch,
        } => commands::describe_inputs(stage, dataset, patch),
        Command::DescribeBackends => commands::describe_backends(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_verify_cli_remains_valid_without_profile() {
        let cli = Cli::try_parse_from(["pystamps", "verify", "--run", "run", "--golden", "golden"])
            .unwrap();
        assert!(matches!(cli.command, Command::Verify { profile: None, .. }));
    }

    #[test]
    fn verify_cli_accepts_scientific_profile_and_stage_scope() {
        let cli = Cli::try_parse_from([
            "pystamps",
            "verify",
            "--run",
            "run",
            "--golden",
            "golden",
            "--profile",
            "scientific",
            "--through-stage",
            "6",
            "--final-products-only",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Command::Verify {
                profile: Some(VerifyProfile::Scientific),
                through_stage: Some(6),
                final_products_only: true,
                ..
            }
        ));
    }
}
