use std::path::Path;

use hdf5_pure::mat::{MatBuilder, Options};
use hdf5_pure::{AttrValue, DType, File};
use num_complex::{Complex32, Complex64};

use super::order::{column_to_row, row_to_column};
use super::{MatArray, MatError, MatFile, MatValue};

pub(super) fn read(path: &Path) -> Result<MatFile, MatError> {
    let file = File::open(path).map_err(hdf_error)?;
    let root = file.root();
    let mut payload = MatFile::new();
    for name in root.datasets().map_err(hdf_error)? {
        let dataset = root.dataset(&name).map_err(hdf_error)?;
        let attrs = dataset.attrs().map_err(hdf_error)?;
        let class = attribute_string(attrs.get("MATLAB_class"))
            .ok_or_else(|| MatError::V73(format!("key {name} has no MATLAB_class")))?;
        let is_empty = attribute_integer(attrs.get("MATLAB_empty")) == Some(1);
        let shape = read_shape(&dataset, is_empty)?;
        payload.insert(
            name.clone(),
            read_dataset(&name, class, shape, is_empty, &dataset)?,
        );
    }
    Ok(payload)
}

fn read_shape(dataset: &hdf5_pure::Dataset<'_>, is_empty: bool) -> Result<Vec<usize>, MatError> {
    let storage_shape = dataset.shape().map_err(hdf_error)?;
    if is_empty && storage_shape.iter().product::<u64>() != 0 {
        return Ok(dataset
            .read_u64()
            .map_err(hdf_error)?
            .into_iter()
            .map(|value| value as usize)
            .collect());
    }
    Ok(storage_shape
        .into_iter()
        .rev()
        .map(|value| value as usize)
        .collect())
}

pub(super) fn write(path: &Path, payload: &MatFile) -> Result<(), MatError> {
    let mut builder = MatBuilder::new(Options::default());
    for (name, value) in payload {
        write_value(&mut builder, name, value)?;
    }
    let bytes = builder
        .finish()
        .map_err(|error| MatError::V73(error.to_string()))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn read_dataset(
    name: &str,
    class: &str,
    shape: Vec<usize>,
    is_empty: bool,
    dataset: &hdf5_pure::Dataset<'_>,
) -> Result<MatValue, MatError> {
    macro_rules! array {
        ($variant:ident, $reader:ident) => {{
            let values = if is_empty {
                Vec::new()
            } else {
                dataset.$reader().map_err(hdf_error)?
            };
            Ok(MatValue::$variant(MatArray {
                values: column_to_row(values, &shape),
                shape,
            }))
        }};
    }
    let dtype = dataset.dtype().map_err(hdf_error)?;
    if let DType::Compound(fields) = dtype {
        return read_complex(name, class, shape, is_empty, dataset, &fields);
    }
    match class {
        "double" => array!(F64, read_f64),
        "single" => array!(F32, read_f32),
        "int64" => array!(I64, read_i64),
        "int32" => array!(I32, read_i32),
        "int16" => array!(I16, read_i16),
        "int8" => array!(I8, read_i8),
        "uint64" => array!(U64, read_u64),
        "uint32" => array!(U32, read_u32),
        "uint16" => array!(U16, read_u16),
        "uint8" => array!(U8, read_u8),
        "logical" => {
            let values = if is_empty {
                Vec::new()
            } else {
                dataset
                    .read_u8()
                    .map_err(hdf_error)?
                    .into_iter()
                    .map(|value| value != 0)
                    .collect()
            };
            Ok(MatValue::Bool(MatArray {
                values: column_to_row(values, &shape),
                shape,
            }))
        }
        "char" => array!(Char, read_u16),
        unsupported => Err(MatError::Unsupported(format!(
            "MAT v7.3 key {name} has class {unsupported}"
        ))),
    }
}

fn read_complex(
    name: &str,
    class: &str,
    shape: Vec<usize>,
    is_empty: bool,
    dataset: &hdf5_pure::Dataset<'_>,
    fields: &[(String, DType)],
) -> Result<MatValue, MatError> {
    if is_empty {
        return match class {
            "double" => Ok(MatValue::ComplexF64(MatArray {
                shape,
                values: Vec::new(),
            })),
            "single" => Ok(MatValue::ComplexF32(MatArray {
                shape,
                values: Vec::new(),
            })),
            _ => Err(MatError::Unsupported(format!(
                "complex class {class} for {name}"
            ))),
        };
    }
    if fields.len() != 2 || fields[0].0 != "real" || fields[1].0 != "imag" {
        return Err(MatError::V73(format!(
            "complex key {name} lacks real/imag fields"
        )));
    }
    let raw = dataset.read_raw().map_err(hdf_error)?;
    let count = shape.iter().product::<usize>();
    match class {
        "double" => {
            if fields[0].1 != DType::F64 || fields[1].1 != DType::F64 || raw.len() != count * 16 {
                return Err(MatError::V73(format!(
                    "complex f64 key {name} layout mismatch"
                )));
            }
            let values = (0..count)
                .map(|index| {
                    let base = index * 16;
                    Complex64::new(read_f64(&raw, base), read_f64(&raw, base + 8))
                })
                .collect();
            Ok(MatValue::ComplexF64(MatArray {
                values: column_to_row(values, &shape),
                shape,
            }))
        }
        "single" => {
            if fields[0].1 != DType::F32 || fields[1].1 != DType::F32 || raw.len() != count * 8 {
                return Err(MatError::V73(format!(
                    "complex f32 key {name} layout mismatch"
                )));
            }
            let values = (0..count)
                .map(|index| {
                    let base = index * 8;
                    Complex32::new(read_f32(&raw, base), read_f32(&raw, base + 4))
                })
                .collect();
            Ok(MatValue::ComplexF32(MatArray {
                values: column_to_row(values, &shape),
                shape,
            }))
        }
        _ => Err(MatError::Unsupported(format!(
            "complex class {class} for {name}"
        ))),
    }
}

fn read_f64(raw: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes(raw[offset..offset + 8].try_into().unwrap())
}

fn read_f32(raw: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(raw[offset..offset + 4].try_into().unwrap())
}

fn write_value(builder: &mut MatBuilder, name: &str, value: &MatValue) -> Result<(), MatError> {
    macro_rules! write {
        ($method:ident, $array:expr) => {{
            let ordered = row_to_column(&$array.values, &$array.shape);
            builder.$method(name, &$array.shape, &ordered)
        }};
    }
    let result = match value {
        MatValue::F64(value) => write!(write_f64, value),
        MatValue::F32(value) => write!(write_f32, value),
        MatValue::I64(value) => write!(write_i64, value),
        MatValue::I32(value) => write!(write_i32, value),
        MatValue::I16(value) => write!(write_i16, value),
        MatValue::I8(value) => write!(write_i8, value),
        MatValue::U64(value) => write!(write_u64, value),
        MatValue::U32(value) => write!(write_u32, value),
        MatValue::U16(value) => write!(write_u16, value),
        MatValue::U8(value) => write!(write_u8, value),
        MatValue::Bool(value) => {
            let ordered: Vec<u8> = row_to_column(&value.values, &value.shape)
                .into_iter()
                .map(u8::from)
                .collect();
            builder.write_logical(name, &value.shape, &ordered)
        }
        MatValue::ComplexF64(value) => {
            let ordered: Vec<_> = row_to_column(&value.values, &value.shape)
                .into_iter()
                .map(|item| (item.re, item.im))
                .collect();
            builder.write_complex_f64(name, &value.shape, &ordered)
        }
        MatValue::ComplexF32(value) => {
            let ordered: Vec<_> = row_to_column(&value.values, &value.shape)
                .into_iter()
                .map(|item| (item.re, item.im))
                .collect();
            builder.write_complex_f32(name, &value.shape, &ordered)
        }
        MatValue::Char(value) => {
            let text = String::from_utf16_lossy(&value.values);
            builder.write_char(name, &text)
        }
        MatValue::Sparse(_) | MatValue::Cell(_) | MatValue::Struct(_) => {
            return Err(MatError::Unsupported(format!(
                "MAT v7.3 write for nested/sparse key {name}"
            )));
        }
    };
    result.map_err(|error| MatError::V73(error.to_string()))?;
    Ok(())
}

fn attribute_string(value: Option<&AttrValue>) -> Option<&str> {
    match value {
        Some(AttrValue::String(value) | AttrValue::AsciiString(value)) => Some(value),
        _ => None,
    }
}

fn attribute_integer(value: Option<&AttrValue>) -> Option<i64> {
    match value {
        Some(AttrValue::I32(value)) => Some(i64::from(*value)),
        Some(AttrValue::I64(value)) => Some(*value),
        Some(AttrValue::U32(value)) => Some(i64::from(*value)),
        Some(AttrValue::U64(value)) => i64::try_from(*value).ok(),
        _ => None,
    }
}

fn hdf_error(error: hdf5_pure::Error) -> MatError {
    MatError::V73(error.to_string())
}
