use num_complex::{Complex32, Complex64};
use pystamps_io::MatArray;
use pystamps_pipeline::config::ToleranceConfig;

use crate::policy::{resolve, NumericPolicy};
use crate::value_compare::compare_layout;
use crate::value_result::{numeric_failure, CompareResult};
use crate::OutlierSummary;

pub(crate) fn compare_real<T: Copy + Into<f64>, U: Copy + Into<f64>>(
    key: &str,
    observed: &MatArray<T>,
    expected: &MatArray<U>,
    tolerance: &ToleranceConfig,
) -> CompareResult {
    compare_real_iter(
        key,
        &observed.shape,
        observed.values.len(),
        observed.values.iter().copied().map(Into::into),
        &expected.shape,
        expected.values.len(),
        expected.values.iter().copied().map(Into::into),
        tolerance,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_real_iter(
    key: &str,
    observed_shape: &[usize],
    observed_len: usize,
    observed: impl Iterator<Item = f64>,
    expected_shape: &[usize],
    expected_len: usize,
    expected: impl Iterator<Item = f64>,
    tolerance: &ToleranceConfig,
) -> CompareResult {
    compare_layout(
        key,
        observed_shape,
        observed_len,
        expected_shape,
        expected_len,
    )?;
    let policy = resolve(key, tolerance);
    let mut max_abs = 0.0_f64;
    let mut outliers = 0_usize;
    let mut compared = 0_usize;
    for (left, right) in observed.zip(expected) {
        if left.is_nan() && right.is_nan() {
            continue;
        }
        compared += 1;
        if left == right {
            continue;
        }
        let nonfinite_mismatch = !left.is_finite() || !right.is_finite();
        let difference = if nonfinite_mismatch {
            f64::INFINITY
        } else if policy.wrapped {
            wrapped_difference(left - right, policy.wrap_period)
        } else {
            (left - right).abs()
        };
        max_abs = max_abs.max(difference);
        let scale = if policy.wrapped { 0.0 } else { right.abs() };
        outliers +=
            usize::from(nonfinite_mismatch || difference > policy.atol + policy.rtol * scale);
    }
    numeric_result(key, compared, outliers, max_abs, policy)
}

pub(crate) fn compare_complex<T, U>(
    key: &str,
    observed: &MatArray<T>,
    expected: &MatArray<U>,
    tolerance: &ToleranceConfig,
) -> CompareResult
where
    T: Copy + ToComplex64,
    U: Copy + ToComplex64,
{
    compare_complex_iter(
        key,
        &observed.shape,
        observed.values.len(),
        observed
            .values
            .iter()
            .copied()
            .map(ToComplex64::to_complex64),
        &expected.shape,
        expected.values.len(),
        expected
            .values
            .iter()
            .copied()
            .map(ToComplex64::to_complex64),
        tolerance,
    )
}

pub(crate) trait ToComplex64 {
    fn to_complex64(self) -> Complex64;
}

impl ToComplex64 for Complex64 {
    fn to_complex64(self) -> Complex64 {
        self
    }
}

impl ToComplex64 for Complex32 {
    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_complex_iter(
    key: &str,
    observed_shape: &[usize],
    observed_len: usize,
    observed: impl Iterator<Item = Complex64>,
    expected_shape: &[usize],
    expected_len: usize,
    expected: impl Iterator<Item = Complex64>,
    tolerance: &ToleranceConfig,
) -> CompareResult {
    compare_layout(
        key,
        observed_shape,
        observed_len,
        expected_shape,
        expected_len,
    )?;
    let policy = resolve(key, tolerance);
    let mut max_abs = 0.0_f64;
    let mut outliers = 0_usize;
    let mut compared = 0_usize;
    for (left, right) in observed.zip(expected) {
        if complex_equal(left, right) {
            if !complex_has_nan(left) {
                compared += 1;
            }
            continue;
        }
        compared += 1;
        let finite = left.re.is_finite()
            && left.im.is_finite()
            && right.re.is_finite()
            && right.im.is_finite();
        let difference = if !finite {
            f64::INFINITY
        } else if policy.wrapped && (left.norm_sqr() == 0.0 || right.norm_sqr() == 0.0) {
            f64::INFINITY
        } else if policy.wrapped {
            wrapped_difference(left.arg() - right.arg(), policy.wrap_period)
        } else {
            (left - right).norm()
        };
        max_abs = max_abs.max(difference);
        let scale = if policy.wrapped { 0.0 } else { right.norm() };
        outliers += usize::from(!finite || difference > policy.atol + policy.rtol * scale);
    }
    numeric_result(key, compared, outliers, max_abs, policy)
}

fn wrapped_difference(value: f64, period: f64) -> f64 {
    ((value + period / 2.0).rem_euclid(period) - period / 2.0).abs()
}

fn complex_equal(left: Complex64, right: Complex64) -> bool {
    (left.re == right.re || (left.re.is_nan() && right.re.is_nan()))
        && (left.im == right.im || (left.im.is_nan() && right.im.is_nan()))
}

fn complex_has_nan(value: Complex64) -> bool {
    value.re.is_nan() || value.im.is_nan()
}

fn numeric_result(
    key: &str,
    total: usize,
    count: usize,
    max_abs: f64,
    policy: NumericPolicy,
) -> CompareResult {
    let fraction = if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    };
    let summary = OutlierSummary {
        key: key.to_owned(),
        count,
        total,
        fraction,
        allowed_fraction: policy.max_outlier_fraction,
        max_abs,
        max_abs_limit: policy.max_abs,
    };
    let exceeds_cap = policy.max_abs.is_some_and(|limit| max_abs > limit);
    if fraction > policy.max_outlier_fraction || exceeds_cap {
        let label = if policy.wrapped {
            "wrap mismatch"
        } else {
            "numeric mismatch"
        };
        Err(numeric_failure(
            key,
            format!(
                "{label}, max_abs={max_abs}, outliers={count}/{total} ({fraction:.6e}), allowed={:.6e}",
                policy.max_outlier_fraction
            ),
            summary,
        ))
    } else if count > 0 {
        Ok(vec![summary])
    } else {
        Ok(Vec::new())
    }
}
