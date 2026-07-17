use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToleranceConfig {
    pub profile: VerificationProfile,
    pub rtol: f64,
    pub atol: f64,
    pub wrap_equivalence: bool,
    pub wrap_period: f64,
    pub wrap_keys: Vec<String>,
    pub exact_keys: Vec<String>,
    pub max_outlier_fraction: f64,
    pub max_abs: Option<f64>,
    pub key_tolerances: BTreeMap<String, KeyToleranceConfig>,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            profile: VerificationProfile::Strict,
            rtol: 1e-5,
            atol: 1e-7,
            wrap_equivalence: true,
            wrap_period: std::f64::consts::TAU,
            wrap_keys: vec!["dph_noise".to_owned()],
            exact_keys: [
                "ix",
                "keep_ix",
                "ix_weed",
                "ix_weed2",
                "ifg_index",
                "master_ix",
                "sort_ix",
                "edgs",
                "n_edge",
                "rowix",
                "colix",
                "n_ps",
                "n_ifg",
                "n_image",
                "nzix",
                "grid_ij",
                "pystamps_input_fingerprint",
                "random_bperp_fingerprint",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            max_outlier_fraction: 0.0,
            max_abs: None,
            key_tolerances: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationProfile {
    #[default]
    Strict,
    Scientific,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KeyToleranceConfig {
    pub rtol: Option<f64>,
    pub atol: Option<f64>,
    pub max_outlier_fraction: Option<f64>,
    pub max_abs: Option<f64>,
}

impl ToleranceConfig {
    pub fn validate(&self) -> Result<(), String> {
        finite_nonnegative("tolerance.rtol", self.rtol)?;
        finite_nonnegative("tolerance.atol", self.atol)?;
        if !self.wrap_period.is_finite() || self.wrap_period <= 0.0 {
            return Err("tolerance.wrap_period must be finite and positive".to_owned());
        }
        fraction("tolerance.max_outlier_fraction", self.max_outlier_fraction)?;
        optional_limit("tolerance.max_abs", self.max_abs)?;
        validate_outlier_policy(
            self.profile,
            "tolerance",
            self.max_outlier_fraction,
            self.max_abs,
        )?;
        for name in self.exact_keys.iter().chain(&self.wrap_keys) {
            if name.trim().is_empty() {
                return Err("tolerance key names must not be empty".to_owned());
            }
        }
        for (key, policy) in &self.key_tolerances {
            self.validate_key_policy(key, policy)?;
        }
        Ok(())
    }

    fn validate_key_policy(&self, key: &str, policy: &KeyToleranceConfig) -> Result<(), String> {
        if key.trim().is_empty() {
            return Err("tolerance.key_tolerances keys must not be empty".to_owned());
        }
        for (field, value) in [("rtol", policy.rtol), ("atol", policy.atol)] {
            if let Some(value) = value {
                finite_nonnegative(&format!("tolerance.key_tolerances.{key}.{field}"), value)?;
            }
        }
        if let Some(value) = policy.max_outlier_fraction {
            fraction(
                &format!("tolerance.key_tolerances.{key}.max_outlier_fraction"),
                value,
            )?;
        }
        optional_limit(
            &format!("tolerance.key_tolerances.{key}.max_abs"),
            policy.max_abs,
        )?;
        validate_outlier_policy(
            self.profile,
            key,
            policy
                .max_outlier_fraction
                .unwrap_or(self.max_outlier_fraction),
            policy.max_abs.or(self.max_abs),
        )
    }
}

fn finite_nonnegative(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be finite and non-negative"))
    }
}

fn fraction(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(format!("{name} must be between zero and one"))
    }
}

fn optional_limit(name: &str, value: Option<f64>) -> Result<(), String> {
    value.map_or(Ok(()), |value| finite_nonnegative(name, value))
}

fn validate_outlier_policy(
    profile: VerificationProfile,
    name: &str,
    allowed: f64,
    cap: Option<f64>,
) -> Result<(), String> {
    if allowed == 0.0 {
        return Ok(());
    }
    if profile == VerificationProfile::Strict {
        return Err(format!(
            "{name}.max_outlier_fraction requires profile=scientific"
        ));
    }
    cap.map(|_| ())
        .ok_or_else(|| format!("{name}.max_abs is required when max_outlier_fraction is non-zero"))
}
