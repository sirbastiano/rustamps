use std::path::Path;

use matrw::{MatVariable, MatlabType, NumericArray, SparseArray};
use num_complex::{Complex32, Complex64};

use super::order::{column_to_row, row_to_column};
use super::{MatArray, MatError, MatFile, MatSparse, MatValue};

pub(super) fn read(path: &Path) -> Result<MatFile, MatError> {
    let source = if has_nonstandard_scipy_header(path)? {
        let mut bytes = std::fs::read(path)?;
        bytes[19] = b',';
        bytes[20] = b' ';
        matrw::load_matfile_from_u8(&bytes)
    } else {
        matrw::load_matfile(&path.to_string_lossy())
    }
    .map_err(|error| MatError::V5(error.to_string()))?;
    source
        .into_iter()
        .map(|(name, value)| Ok((name.clone(), decode_value(&name, value)?)))
        .collect()
}

fn has_nonstandard_scipy_header(path: &Path) -> Result<bool, MatError> {
    use std::io::Read;

    let mut header = [0_u8; 21];
    std::fs::File::open(path)?.read_exact(&mut header)?;
    Ok(header.starts_with(b"MATLAB 5.0 MAT-file") && header[19] != b',')
}

pub(super) fn write(path: &Path, payload: &MatFile) -> Result<(), MatError> {
    let mut destination = matrw::MatFile::new();
    for (name, value) in payload {
        destination.insert(name, encode_value(name, value)?);
    }
    matrw::save_matfile_v7(&path.to_string_lossy(), destination, false)
        .map_err(|error| MatError::V5(error.to_string()))
}

fn decode_value(name: &str, value: MatVariable) -> Result<MatValue, MatError> {
    match value {
        MatVariable::NumericArray(array) => decode_numeric(name, array),
        MatVariable::SparseArray(array) => decode_sparse(name, array),
        other => Err(MatError::Unsupported(format!(
            "MAT v5 key {name} uses unsupported value {other:?}"
        ))),
    }
}

fn decode_numeric(name: &str, array: NumericArray) -> Result<MatValue, MatError> {
    let shape = array.dim;
    let imag = array.value_cmp;
    macro_rules! real {
        ($variant:ident, $values:expr) => {
            Ok(MatValue::$variant(MatArray {
                shape: shape.clone(),
                values: column_to_row($values, &shape),
            }))
        };
    }
    match (array.value, imag) {
        (MatlabType::F64(real), Some(MatlabType::F64(imag))) => {
            let values = real
                .into_iter()
                .zip(imag)
                .map(|(re, im)| Complex64::new(re, im))
                .collect();
            real!(ComplexF64, values)
        }
        (MatlabType::F32(real), Some(MatlabType::F32(imag))) => {
            let values = real
                .into_iter()
                .zip(imag)
                .map(|(re, im)| Complex32::new(re, im))
                .collect();
            real!(ComplexF32, values)
        }
        (MatlabType::F64(values), None) => real!(F64, values),
        (MatlabType::F32(values), None) => real!(F32, values),
        (MatlabType::I64(values), None) => real!(I64, values),
        (MatlabType::I32(values), None) => real!(I32, values),
        (MatlabType::I16(values), None) => real!(I16, values),
        (MatlabType::I8(values), None) => real!(I8, values),
        (MatlabType::U64(values), None) => real!(U64, values),
        (MatlabType::U32(values), None) => real!(U32, values),
        (MatlabType::U16(values), None) => real!(U16, values),
        (MatlabType::U8(values), None) => real!(U8, values),
        (MatlabType::BOOL(values), None) => real!(Bool, values),
        (MatlabType::UTF8(values), None) | (MatlabType::UTF16(values), None) => {
            let mut units = Vec::new();
            for character in values {
                let mut encoded = [0_u16; 2];
                units.extend_from_slice(character.encode_utf16(&mut encoded));
            }
            real!(Char, units)
        }
        (kind, component) => Err(MatError::Unsupported(format!(
            "MAT v5 key {name} has incompatible numeric data {kind:?}/{component:?}"
        ))),
    }
}

fn decode_sparse(name: &str, array: SparseArray) -> Result<MatValue, MatError> {
    let imag = match array.value_cmp {
        Some(MatlabType::F64(values)) => Some(values),
        Some(other) => {
            return Err(MatError::Unsupported(format!(
                "MAT v5 sparse key {name} has imag {other:?}"
            )))
        }
        None => None,
    };
    let real = match array.value {
        MatlabType::F64(values) => values,
        MatlabType::BOOL(values) => values.into_iter().map(u8::from).map(f64::from).collect(),
        other => {
            return Err(MatError::Unsupported(format!(
                "MAT v5 sparse key {name} has data {other:?}"
            )))
        }
    };
    Ok(MatValue::Sparse(MatSparse {
        rows: array.dim[0],
        cols: array.dim[1],
        row_indices: array.ir,
        column_offsets: array.jc,
        real,
        imag,
    }))
}

fn encode_value(name: &str, value: &MatValue) -> Result<MatVariable, MatError> {
    macro_rules! dense {
        ($array:expr) => {{
            let value = MatlabType::from(row_to_column(&$array.values, &$array.shape));
            NumericArray::new($array.shape.clone(), value, None)
        }};
    }
    let array = match value {
        MatValue::F64(value) => dense!(value),
        MatValue::F32(value) => dense!(value),
        MatValue::I64(value) => dense!(value),
        MatValue::I32(value) => dense!(value),
        MatValue::I16(value) => dense!(value),
        MatValue::I8(value) => dense!(value),
        MatValue::U64(value) => dense!(value),
        MatValue::U32(value) => dense!(value),
        MatValue::U16(value) => dense!(value),
        MatValue::U8(value) => dense!(value),
        MatValue::Bool(value) => dense!(value),
        MatValue::Char(_) => {
            return Err(MatError::Unsupported(format!(
                "MAT v5 write for char key {name}; use MatFormat::V73"
            )));
        }
        MatValue::ComplexF64(value) => {
            let ordered = row_to_column(&value.values, &value.shape);
            NumericArray::new(
                value.shape.clone(),
                MatlabType::F64(ordered.iter().map(|item| item.re).collect()),
                Some(MatlabType::F64(
                    ordered.iter().map(|item| item.im).collect(),
                )),
            )
        }
        MatValue::ComplexF32(value) => {
            let ordered = row_to_column(&value.values, &value.shape);
            NumericArray::new(
                value.shape.clone(),
                MatlabType::F32(ordered.iter().map(|item| item.re).collect()),
                Some(MatlabType::F32(
                    ordered.iter().map(|item| item.im).collect(),
                )),
            )
        }
        MatValue::Sparse(value) => {
            let sparse = SparseArray::new(
                value.rows,
                value.cols,
                value.row_indices.clone(),
                value.column_offsets.clone(),
                MatlabType::F64(value.real.clone()),
                value.imag.clone().map(MatlabType::F64),
            )
            .map_err(|error| MatError::V5(error.to_string()))?;
            return Ok(MatVariable::SparseArray(sparse));
        }
        MatValue::Cell(_) | MatValue::Struct(_) => {
            return Err(MatError::Unsupported(format!(
                "MAT v5 write for nested key {name}"
            )));
        }
    }
    .map_err(|error| MatError::V5(error.to_string()))?;
    Ok(MatVariable::NumericArray(array))
}
