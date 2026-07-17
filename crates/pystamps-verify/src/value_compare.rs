use num_complex::Complex64;
use pystamps_io::mat::MatSparse;
use pystamps_io::{MatFile, MatValue};
use pystamps_pipeline::config::ToleranceConfig;

use crate::value_numeric::{
    compare_complex, compare_complex_iter, compare_real, compare_real_iter,
};
use crate::value_result::{failure, CompareFailure, CompareResult};
use crate::value_support::{compare_exact, kind};
use crate::FileComparison;

pub(crate) fn compare_file(
    relative: &str,
    run: &MatFile,
    golden: &MatFile,
    tolerance: &ToleranceConfig,
) -> FileComparison {
    let mut outliers = Vec::new();
    for (key, expected) in golden {
        let Some(observed) = run.get(key) else {
            return failed(relative, failure(key, "missing key", None));
        };
        match compare_value(key, observed, expected, tolerance) {
            Ok(mut found) => outliers.append(&mut found),
            Err(error) => return failed(relative, error),
        }
    }
    let message = if outliers.is_empty() {
        format!("matched {} keys", golden.len())
    } else {
        format!(
            "matched {} keys with bounded outliers in {} arrays",
            golden.len(),
            outliers.len()
        )
    };
    FileComparison {
        path: relative.to_owned(),
        ok: true,
        message,
        failing_key: None,
        max_abs: None,
        outliers,
    }
}

fn compare_value(
    key: &str,
    observed: &MatValue,
    expected: &MatValue,
    tolerance: &ToleranceConfig,
) -> CompareResult {
    macro_rules! exact {
        ($left:expr, $right:expr) => {
            compare_exact(key, $left, $right)
        };
    }
    macro_rules! real {
        ($left:expr, $right:expr) => {
            compare_real(key, $left, $right, tolerance)
        };
    }
    macro_rules! complex {
        ($left:expr, $right:expr) => {
            compare_complex(key, $left, $right, tolerance)
        };
    }
    match (observed, expected) {
        (MatValue::F64(a), MatValue::F64(b)) => real!(a, b),
        (MatValue::F32(a), MatValue::F32(b)) => real!(a, b),
        (MatValue::F32(a), MatValue::F64(b)) => compare_real_iter(
            key,
            &a.shape,
            a.values.len(),
            a.values.iter().map(|&v| f64::from(v)),
            &b.shape,
            b.values.len(),
            b.values.iter().copied(),
            tolerance,
        ),
        (MatValue::F64(a), MatValue::F32(b)) => compare_real_iter(
            key,
            &a.shape,
            a.values.len(),
            a.values.iter().copied(),
            &b.shape,
            b.values.len(),
            b.values.iter().map(|&v| f64::from(v)),
            tolerance,
        ),
        (MatValue::ComplexF64(a), MatValue::ComplexF64(b)) => complex!(a, b),
        (MatValue::ComplexF32(a), MatValue::ComplexF32(b)) => complex!(a, b),
        (MatValue::ComplexF32(a), MatValue::ComplexF64(b)) => compare_complex_iter(
            key,
            &a.shape,
            a.values.len(),
            a.values
                .iter()
                .map(|v| Complex64::new(v.re.into(), v.im.into())),
            &b.shape,
            b.values.len(),
            b.values.iter().copied(),
            tolerance,
        ),
        (MatValue::ComplexF64(a), MatValue::ComplexF32(b)) => compare_complex_iter(
            key,
            &a.shape,
            a.values.len(),
            a.values.iter().copied(),
            &b.shape,
            b.values.len(),
            b.values
                .iter()
                .map(|v| Complex64::new(v.re.into(), v.im.into())),
            tolerance,
        ),
        (MatValue::I64(a), MatValue::I64(b)) => exact!(a, b),
        (MatValue::I32(a), MatValue::I32(b)) => exact!(a, b),
        (MatValue::I16(a), MatValue::I16(b)) => exact!(a, b),
        (MatValue::I8(a), MatValue::I8(b)) => exact!(a, b),
        (MatValue::U64(a), MatValue::U64(b)) => exact!(a, b),
        (MatValue::U32(a), MatValue::U32(b)) => exact!(a, b),
        (MatValue::U16(a), MatValue::U16(b)) => exact!(a, b),
        (MatValue::U8(a), MatValue::U8(b)) => exact!(a, b),
        (MatValue::Bool(a), MatValue::Bool(b)) => exact!(a, b),
        (MatValue::Char(a), MatValue::Char(b)) => exact!(a, b),
        (MatValue::Sparse(a), MatValue::Sparse(b)) => compare_sparse(key, a, b, tolerance),
        (MatValue::Cell(a), MatValue::Cell(b)) => compare_cell(key, a, b, tolerance),
        (MatValue::Struct(a), MatValue::Struct(b)) => {
            let mut outliers = Vec::new();
            for (field, expected) in b {
                let nested = format!("{key}.{field}");
                let Some(observed) = a.get(field) else {
                    return Err(failure(&nested, "missing field", None));
                };
                outliers.extend(compare_value(&nested, observed, expected, tolerance)?);
            }
            Ok(outliers)
        }
        _ => Err(failure(
            key,
            &format!("type mismatch: {} != {}", kind(observed), kind(expected)),
            None,
        )),
    }
}

fn compare_sparse(
    key: &str,
    observed: &MatSparse,
    expected: &MatSparse,
    tolerance: &ToleranceConfig,
) -> CompareResult {
    if (observed.rows, observed.cols) != (expected.rows, expected.cols) {
        return Err(failure(key, "sparse shape mismatch", None));
    }
    compare_slice(
        &format!("{key}.ir"),
        &observed.row_indices,
        &expected.row_indices,
    )?;
    compare_slice(
        &format!("{key}.jc"),
        &observed.column_offsets,
        &expected.column_offsets,
    )?;
    let data_key = format!("{key}.data");
    if observed.real.len() != expected.real.len()
        || observed
            .imag
            .as_ref()
            .is_some_and(|v| v.len() != observed.real.len())
        || expected
            .imag
            .as_ref()
            .is_some_and(|v| v.len() != expected.real.len())
    {
        return Err(failure(&data_key, "sparse data length mismatch", None));
    }
    if observed.imag.is_none() && expected.imag.is_none() {
        return compare_real_iter(
            &data_key,
            &[observed.real.len()],
            observed.real.len(),
            observed.real.iter().copied(),
            &[expected.real.len()],
            expected.real.len(),
            expected.real.iter().copied(),
            tolerance,
        );
    }
    let left = (0..observed.real.len()).map(|index| {
        Complex64::new(
            observed.real[index],
            observed.imag.as_ref().map_or(0.0, |imag| imag[index]),
        )
    });
    let right = (0..expected.real.len()).map(|index| {
        Complex64::new(
            expected.real[index],
            expected.imag.as_ref().map_or(0.0, |imag| imag[index]),
        )
    });
    compare_complex_iter(
        &data_key,
        &[observed.real.len()],
        observed.real.len(),
        left,
        &[expected.real.len()],
        expected.real.len(),
        right,
        tolerance,
    )
}

fn compare_cell(
    key: &str,
    observed: &[MatValue],
    expected: &[MatValue],
    tolerance: &ToleranceConfig,
) -> CompareResult {
    if observed.len() != expected.len() {
        return Err(failure(key, "cell length mismatch", None));
    }
    let mut outliers = Vec::new();
    for (index, (left, right)) in observed.iter().zip(expected).enumerate() {
        outliers.extend(compare_value(
            &format!("{key}[{index}]"),
            left,
            right,
            tolerance,
        )?);
    }
    Ok(outliers)
}

fn compare_slice<T: PartialEq>(
    key: &str,
    observed: &[T],
    expected: &[T],
) -> Result<(), CompareFailure> {
    if observed == expected {
        Ok(())
    } else {
        Err(failure(key, "exact value mismatch", None))
    }
}

pub(crate) fn compare_layout(
    key: &str,
    observed_shape: &[usize],
    observed_len: usize,
    expected_shape: &[usize],
    expected_len: usize,
) -> Result<(), CompareFailure> {
    if observed_shape != expected_shape {
        return Err(failure(
            key,
            &format!("shape mismatch {observed_shape:?} != {expected_shape:?}"),
            None,
        ));
    }
    if observed_len != expected_len {
        return Err(failure(
            key,
            &format!("element count mismatch {observed_len} != {expected_len}"),
            None,
        ));
    }
    Ok(())
}

fn failed(path: &str, error: CompareFailure) -> FileComparison {
    FileComparison {
        path: path.to_owned(),
        ok: false,
        message: error.message,
        failing_key: Some(error.key),
        max_abs: error.max_abs,
        outliers: error.outliers,
    }
}

#[cfg(test)]
#[path = "value_compare_tests.rs"]
mod tests;
