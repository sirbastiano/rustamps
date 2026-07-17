use std::path::Path;

use rustamps_core::stages::stage2::{random_coherence_histogram, PsquareReference};
use rustamps_io::read_mat;

use super::super::mat::numeric_f64;

const RANDOM_SEED: u32 = 2005;
const RANDOM_SAMPLES: usize = 300_000;

pub struct Reference {
    pub model: PsquareReference,
    pub cache_hit: bool,
    pub bperp_fingerprint: u64,
}

pub fn coherence_bins() -> Vec<f64> {
    (0..100).map(|index| 0.005 + index as f64 * 0.01).collect()
}

pub fn load_or_generate(
    patch: &Path,
    bperp: &[f64],
    n_trial_wraps: f64,
) -> Result<Reference, String> {
    let bins = coherence_bins();
    let fingerprint = bperp_fingerprint(bperp);
    if let Some((distribution, last)) = load_pm_cache(patch, &bins, n_trial_wraps, fingerprint) {
        return Ok(Reference {
            model: PsquareReference {
                coherence_bins: bins,
                random_distribution: distribution,
                low_coherence_bins: 31,
                last_nonzero_random_bin_one_based: last,
            },
            cache_hit: true,
            bperp_fingerprint: fingerprint,
        });
    }
    let distribution =
        random_coherence_histogram(RANDOM_SEED, RANDOM_SAMPLES, bperp, n_trial_wraps, &bins)
            .map_err(|error| error.to_string())?;
    let last = distribution
        .iter()
        .rposition(|value| *value > 0.0)
        .map_or(1, |index| index + 1);
    Ok(Reference {
        model: PsquareReference {
            coherence_bins: bins,
            random_distribution: distribution,
            low_coherence_bins: 31,
            last_nonzero_random_bin_one_based: last,
        },
        cache_hit: false,
        bperp_fingerprint: fingerprint,
    })
}

pub fn signal_noise_schema(bperp: &[f64]) -> Reference {
    let bins = coherence_bins();
    Reference {
        model: PsquareReference {
            random_distribution: vec![0.0; bins.len()],
            coherence_bins: bins,
            low_coherence_bins: 31,
            last_nonzero_random_bin_one_based: 1,
        },
        cache_hit: false,
        bperp_fingerprint: bperp_fingerprint(bperp),
    }
}

pub(super) fn load_pm_cache(
    patch: &Path,
    bins: &[f64],
    n_trial_wraps: f64,
    fingerprint: u64,
) -> Option<(Vec<f64>, usize)> {
    let file = read_mat(patch.join("pm1.mat")).ok()?;
    let saved_bins = numeric_f64(&file, "coh_bins").ok()?;
    let distribution = numeric_f64(&file, "Nr").ok()?;
    let saved_wraps = *numeric_f64(&file, "n_trial_wraps").ok()?.first()?;
    let last = *numeric_f64(&file, "Nr_max_nz_ix").ok()?.first()?;
    let saved_fingerprint = *numeric_f64(&file, "random_bperp_fingerprint")
        .ok()?
        .first()?;
    let expected_f32 = f64::from(n_trial_wraps as f32);
    if saved_bins.len() != bins.len()
        || distribution.len() != bins.len()
        || !saved_bins
            .iter()
            .zip(bins)
            .all(|(saved, expected)| (saved - expected).abs() <= 1e-12)
        || (!close(saved_wraps, n_trial_wraps) && !close(saved_wraps, expected_f32))
        || distribution
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || distribution.iter().all(|value| *value == 0.0)
        || saved_fingerprint != fingerprint as f64
        || !last.is_finite()
        || last.fract() != 0.0
        || !(1.0..=bins.len() as f64).contains(&last)
    {
        return None;
    }
    Some((distribution, last as usize))
}

pub(super) fn bperp_fingerprint(values: &[f64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in std::iter::once(values.len() as u64).chain(values.iter().map(|v| v.to_bits())) {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    (hash & ((1_u64 << 52) - 1)).max(1)
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12
}

#[cfg(test)]
pub(super) const fn random_samples() -> usize {
    RANDOM_SAMPLES
}
