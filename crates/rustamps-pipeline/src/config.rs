use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[path = "config_native.rs"]
mod config_native;
#[path = "config_tolerance.rs"]
mod config_tolerance;

pub use config_tolerance::{KeyToleranceConfig, ToleranceConfig, VerificationProfile};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub io_workers: usize,
    pub cpu_workers: usize,
    pub backend: String,
    pub stage2_kernel_backend: String,
    pub stage2_native_threads: usize,
    pub stage6_solver: String,
    pub stage6_grid_scale: f64,
    pub stage6_max_flow_passes: usize,
    pub stage6_ifg_workers: usize,
    pub stage7_chunk_ps: usize,
    pub stage8_chunk_edges: usize,
    pub enable_mat_stage_cache: bool,
    pub stage2_checkpoint_mode: String,
    pub stage2_checkpoint_interval: usize,
    pub stage2_debug: bool,
    pub stage4_debug: bool,
    pub kernel_backend_overrides: BTreeMap<String, String>,
    pub stage2_patch_backend_overrides: BTreeMap<String, String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            io_workers: 8,
            cpu_workers: 0,
            backend: "native".to_owned(),
            stage2_kernel_backend: "native".to_owned(),
            stage2_native_threads: 0,
            stage6_solver: "native".to_owned(),
            stage6_grid_scale: 1.0,
            stage6_max_flow_passes: 0,
            stage6_ifg_workers: 0,
            stage7_chunk_ps: 100_000,
            stage8_chunk_edges: 200_000,
            enable_mat_stage_cache: true,
            stage2_checkpoint_mode: "final".to_owned(),
            stage2_checkpoint_interval: 1,
            stage2_debug: false,
            stage4_debug: false,
            kernel_backend_overrides: BTreeMap::new(),
            stage2_patch_backend_overrides: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompatibilityConfig {
    pub reference_root: Option<String>,
    pub strict_reference: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    pub runtime: RuntimeConfig,
    pub tolerance: ToleranceConfig,
    pub compatibility: CompatibilityConfig,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("configuration I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("configuration parse failed: {0}")]
    Parse(String),
    #[error("unsupported native configuration: {0}")]
    Unsupported(String),
}

pub fn load_config(path: Option<&Path>) -> Result<RunConfig, ConfigError> {
    load_config_with_profile(path, None)
}

pub fn load_config_with_profile(
    path: Option<&Path>,
    profile: Option<VerificationProfile>,
) -> Result<RunConfig, ConfigError> {
    let mut config = match path {
        None => RunConfig::default(),
        Some(path) => {
            let text = fs::read_to_string(path)?;
            if path.extension().and_then(|value| value.to_str()) == Some("json") {
                serde_json::from_str(&text)
                    .map_err(|error| ConfigError::Parse(error.to_string()))?
            } else {
                serde_yaml_ng::from_str(&text)
                    .map_err(|error| ConfigError::Parse(error.to_string()))?
            }
        }
    };
    apply_profile_and_normalize(&mut config, profile)?;
    Ok(config)
}

fn apply_profile_and_normalize(
    config: &mut RunConfig,
    profile: Option<VerificationProfile>,
) -> Result<(), ConfigError> {
    if let Some(profile) = profile {
        config.tolerance.profile = profile;
    }
    normalize(config)
}

fn normalize(config: &mut RunConfig) -> Result<(), ConfigError> {
    config.runtime.backend = native_alias(&config.runtime.backend, "runtime.backend")?;
    config.runtime.stage2_kernel_backend = native_alias(
        &config.runtime.stage2_kernel_backend,
        "runtime.stage2_kernel_backend",
    )?;
    config.runtime.stage6_solver = match config
        .runtime
        .stage6_solver
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "auto" | "native" | "backend" => "native".to_owned(),
        "snaphu" | "external" => {
            return Err(ConfigError::Unsupported(
                "stage6_solver external/SNAPHU was removed; use native".to_owned(),
            ));
        }
        value => return Err(ConfigError::Unsupported(format!("stage6_solver={value}"))),
    };
    if !config.runtime.stage6_grid_scale.is_finite() || config.runtime.stage6_grid_scale <= 0.0 {
        return Err(ConfigError::Unsupported(
            "stage6_grid_scale must be finite and positive".to_owned(),
        ));
    }
    if config.runtime.stage6_max_flow_passes != 0 {
        return Err(ConfigError::Unsupported(
            "bounded Stage 6 flow passes failed scientific validation; use 0 (converged)"
                .to_owned(),
        ));
    }
    if !matches!(config.runtime.stage6_ifg_workers, 0 | 1 | 2 | 4) {
        return Err(ConfigError::Unsupported(
            "stage6_ifg_workers must be 0 (auto), 1, 2, or 4".to_owned(),
        ));
    }
    config_native::reject_inert_options(config)?;
    config
        .tolerance
        .validate()
        .map_err(ConfigError::Unsupported)?;
    Ok(())
}

fn native_alias(value: &str, field: &str) -> Result<String, ConfigError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" | "native" => Ok("native".to_owned()),
        unsupported => Err(ConfigError::Unsupported(format!(
            "{field}={unsupported}; only native is available"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_stage6_solver_is_rejected() {
        let mut config = RunConfig::default();
        config.runtime.stage6_solver = "snaphu".to_owned();
        assert!(normalize(&mut config)
            .unwrap_err()
            .to_string()
            .contains("removed"));
    }

    #[test]
    fn stage6_grid_scale_must_be_finite_and_positive() {
        for value in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let mut config = RunConfig::default();
            config.runtime.stage6_grid_scale = value;
            assert!(normalize(&mut config).is_err());
        }
    }

    #[test]
    fn bounded_stage6_flow_is_rejected() {
        let mut config = RunConfig::default();
        config.runtime.stage6_max_flow_passes = 1;
        assert!(normalize(&mut config)
            .unwrap_err()
            .to_string()
            .contains("failed scientific validation"));
    }

    #[test]
    fn stage6_ifg_workers_accepts_auto_and_benchmark_sizes() {
        for workers in [0, 1, 2, 4] {
            let mut config = RunConfig::default();
            config.runtime.stage6_ifg_workers = workers;
            assert!(normalize(&mut config).is_ok());
        }
        let mut config = RunConfig::default();
        config.runtime.stage6_ifg_workers = 3;
        assert!(normalize(&mut config)
            .unwrap_err()
            .to_string()
            .contains("0 (auto), 1, 2, or 4"));
    }

    #[test]
    fn legacy_external_tool_paths_are_not_accepted() {
        let error =
            serde_yaml_ng::from_str::<RunConfig>("tools:\n  snaphu: /tmp/snaphu\n").unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn legacy_tolerance_yaml_defaults_to_strict() {
        let config: RunConfig = serde_yaml_ng::from_str(
            "tolerance:\n  rtol: 0.001\n  atol: 0.0001\n  wrap_keys: [dph_noise]\n",
        )
        .unwrap();
        assert_eq!(config.tolerance.profile, VerificationProfile::Strict);
        assert_eq!(config.tolerance.rtol, 0.001);
    }

    #[test]
    fn scientific_outliers_require_a_hard_cap() {
        let mut strict = RunConfig::default();
        strict.tolerance.max_outlier_fraction = 0.01;
        strict.tolerance.max_abs = Some(0.1);
        assert!(normalize(&mut strict)
            .unwrap_err()
            .to_string()
            .contains("profile=scientific"));

        let mut config = RunConfig::default();
        config.tolerance.profile = VerificationProfile::Scientific;
        config.tolerance.max_outlier_fraction = 0.01;
        assert!(normalize(&mut config)
            .unwrap_err()
            .to_string()
            .contains("max_abs"));
        config.tolerance.max_abs = Some(0.1);
        assert!(normalize(&mut config).is_ok());
    }

    #[test]
    fn scientific_profile_deserializes_from_yaml() {
        let config: RunConfig =
            serde_yaml_ng::from_str("tolerance:\n  profile: scientific\n").unwrap();
        assert_eq!(config.tolerance.profile, VerificationProfile::Scientific);
    }

    #[test]
    fn profile_override_is_applied_before_tolerance_validation() {
        let mut config = RunConfig::default();
        config.tolerance.max_outlier_fraction = 0.01;
        config.tolerance.max_abs = Some(0.1);
        let mut strict = config.clone();
        assert!(normalize(&mut strict).is_err());
        assert!(
            apply_profile_and_normalize(&mut config, Some(VerificationProfile::Scientific)).is_ok()
        );
    }
}
