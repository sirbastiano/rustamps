use std::fmt::Debug;

use rustamps_io::{MatArray, MatValue};

use crate::value_compare::compare_layout;
use crate::value_result::{failure, CompareResult};

pub(crate) fn compare_exact<T: PartialEq + Debug>(
    key: &str,
    observed: &MatArray<T>,
    expected: &MatArray<T>,
) -> CompareResult {
    compare_layout(
        key,
        &observed.shape,
        observed.values.len(),
        &expected.shape,
        expected.values.len(),
    )?;
    if observed.values == expected.values {
        Ok(Vec::new())
    } else {
        Err(failure(key, "exact value mismatch", None))
    }
}

pub(crate) fn kind(value: &MatValue) -> &'static str {
    match value {
        MatValue::F64(_) => "f64",
        MatValue::F32(_) => "f32",
        MatValue::I64(_) => "i64",
        MatValue::I32(_) => "i32",
        MatValue::I16(_) => "i16",
        MatValue::I8(_) => "i8",
        MatValue::U64(_) => "u64",
        MatValue::U32(_) => "u32",
        MatValue::U16(_) => "u16",
        MatValue::U8(_) => "u8",
        MatValue::Bool(_) => "logical",
        MatValue::ComplexF64(_) => "complex f64",
        MatValue::ComplexF32(_) => "complex f32",
        MatValue::Char(_) => "char",
        MatValue::Sparse(_) => "sparse",
        MatValue::Cell(_) => "cell",
        MatValue::Struct(_) => "struct",
    }
}
