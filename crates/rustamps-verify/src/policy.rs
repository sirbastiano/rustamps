use rustamps_pipeline::config::{ToleranceConfig, VerificationProfile};

const SCIENTIFIC_WRAP_KEYS: &[&str] = &["C_ps", "C_ps2", "ph_res", "ph_res2"];

#[derive(Debug, Clone, Copy)]
pub(crate) struct NumericPolicy {
    pub rtol: f64,
    pub atol: f64,
    pub wrapped: bool,
    pub wrap_period: f64,
    pub max_outlier_fraction: f64,
    pub max_abs: Option<f64>,
}

pub(crate) fn resolve(key: &str, tolerance: &ToleranceConfig) -> NumericPolicy {
    if matches_any(key, &tolerance.exact_keys) {
        return NumericPolicy {
            rtol: 0.0,
            atol: 0.0,
            wrapped: false,
            wrap_period: tolerance.wrap_period,
            max_outlier_fraction: 0.0,
            max_abs: Some(0.0),
        };
    }
    let override_policy = tolerance
        .key_tolerances
        .iter()
        .filter(|(name, _)| matches_key(key, name))
        .max_by_key(|(name, _)| name.len())
        .map(|(_, policy)| policy);
    let configured_wrap = matches_any(key, &tolerance.wrap_keys);
    let scientific_wrap = tolerance.profile == VerificationProfile::Scientific
        && SCIENTIFIC_WRAP_KEYS
            .iter()
            .any(|name| matches_key(key, name));
    NumericPolicy {
        rtol: override_policy
            .and_then(|policy| policy.rtol)
            .unwrap_or(tolerance.rtol),
        atol: override_policy
            .and_then(|policy| policy.atol)
            .unwrap_or(tolerance.atol),
        wrapped: tolerance.wrap_equivalence && (configured_wrap || scientific_wrap),
        wrap_period: tolerance.wrap_period,
        max_outlier_fraction: override_policy
            .and_then(|policy| policy.max_outlier_fraction)
            .unwrap_or(tolerance.max_outlier_fraction),
        max_abs: override_policy
            .and_then(|policy| policy.max_abs)
            .or(tolerance.max_abs),
    }
}

fn matches_any(key: &str, names: &[String]) -> bool {
    names.iter().any(|name| matches_key(key, name))
}

fn matches_key(key: &str, name: &str) -> bool {
    key == name
        || key
            .strip_suffix(name)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rustamps_pipeline::config::KeyToleranceConfig;

    use super::*;

    #[test]
    fn exact_keys_win_and_longest_suffix_override_is_used() {
        let mut tolerance = ToleranceConfig::default();
        tolerance.key_tolerances = BTreeMap::from([
            (
                "value".to_owned(),
                KeyToleranceConfig {
                    atol: Some(1.0),
                    ..KeyToleranceConfig::default()
                },
            ),
            (
                "nested.value".to_owned(),
                KeyToleranceConfig {
                    atol: Some(2.0),
                    ..KeyToleranceConfig::default()
                },
            ),
        ]);
        assert_eq!(resolve("root.nested.value", &tolerance).atol, 2.0);
        assert_eq!(resolve("root.ix", &tolerance).atol, 0.0);
    }

    #[test]
    fn scientific_profile_wraps_residual_but_not_unwrapped_phase() {
        let mut tolerance = ToleranceConfig::default();
        tolerance.profile = VerificationProfile::Scientific;
        assert!(resolve("ph_res", &tolerance).wrapped);
        assert!(!resolve("ph_uw", &tolerance).wrapped);
    }
}
