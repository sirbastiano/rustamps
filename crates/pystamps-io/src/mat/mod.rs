mod order;
mod v5;
mod v73;
mod value;

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

pub use value::{MatArray, MatFile, MatSparse, MatValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatFormat {
    Level5,
    V73,
}

#[derive(Debug, Error)]
pub enum MatError {
    #[error("MAT I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported or corrupt MAT file: {0}")]
    Unsupported(String),
    #[error("MAT array shape does not match its values for key {0}")]
    Shape(String),
    #[error("raw byte length {0} is not divisible by element size {1}")]
    RawSize(usize, usize),
    #[error("MAT v5 error: {0}")]
    V5(String),
    #[error("MAT v7.3 error: {0}")]
    V73(String),
}

pub fn detect_format(path: impl AsRef<Path>) -> Result<MatFormat, MatError> {
    let mut header = [0_u8; 128];
    File::open(path)?.read_exact(&mut header)?;
    let text = String::from_utf8_lossy(&header[..116]);
    if text.contains("MATLAB 7.3 MAT-file") {
        Ok(MatFormat::V73)
    } else if text.contains("MATLAB 5.0 MAT-file") {
        Ok(MatFormat::Level5)
    } else {
        Err(MatError::Unsupported(
            text.trim_end_matches('\0').trim().to_owned(),
        ))
    }
}

pub fn read_mat(path: impl AsRef<Path>) -> Result<MatFile, MatError> {
    let path = path.as_ref();
    match detect_format(path)? {
        MatFormat::Level5 => v5::read(path),
        MatFormat::V73 => v73::read(path),
    }
}

pub fn write_mat(path: impl AsRef<Path>, payload: &MatFile) -> Result<(), MatError> {
    let needs_v73 = payload
        .values()
        .any(|value| value_exceeds_level5(value) || value_requires_v73(value));
    if needs_v73 {
        write_mat_with_format(path, payload, MatFormat::V73)
    } else {
        write_mat_with_format(path, payload, MatFormat::Level5)
    }
}

pub fn write_mat_with_format(
    path: impl AsRef<Path>,
    payload: &MatFile,
    format: MatFormat,
) -> Result<(), MatError> {
    let path = path.as_ref();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let backup = parent.join(format!(".{name}.bak-{}-{nonce}", std::process::id()));
    let result = match format {
        MatFormat::Level5 => v5::write(&temporary, payload),
        MatFormat::V73 => v73::write(&temporary, payload),
    };
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    let had_destination = path.exists();
    if had_destination {
        if let Err(error) = fs::rename(path, &backup) {
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
    }
    if let Err(error) = fs::rename(&temporary, path) {
        if had_destination {
            let _ = fs::rename(&backup, path);
        }
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    if had_destination {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

fn value_requires_v73(value: &MatValue) -> bool {
    // matrw encodes every empty Level-5 numeric array as uint8, losing the
    // declared MATLAB class.  V7.3 preserves both the type and empty shape.
    if value.shape().is_some_and(|shape| shape.contains(&0)) {
        return true;
    }
    match value {
        MatValue::Char(_) => true,
        MatValue::Cell(values) => values.iter().any(value_requires_v73),
        MatValue::Struct(values) => values.values().any(value_requires_v73),
        _ => false,
    }
}

fn value_exceeds_level5(value: &MatValue) -> bool {
    const LIMIT: usize = i32::MAX as usize;
    let bytes = match value {
        MatValue::F64(v) => v.values.len().checked_mul(8),
        MatValue::F32(v) => v.values.len().checked_mul(4),
        MatValue::I64(v) => v.values.len().checked_mul(8),
        MatValue::I32(v) => v.values.len().checked_mul(4),
        MatValue::I16(v) => v.values.len().checked_mul(2),
        MatValue::I8(v) => Some(v.values.len()),
        MatValue::U64(v) => v.values.len().checked_mul(8),
        MatValue::U32(v) => v.values.len().checked_mul(4),
        MatValue::U16(v) | MatValue::Char(v) => v.values.len().checked_mul(2),
        MatValue::U8(v) => Some(v.values.len()),
        MatValue::Bool(v) => Some(v.values.len()),
        MatValue::ComplexF64(v) => v.values.len().checked_mul(16),
        MatValue::ComplexF32(v) => v.values.len().checked_mul(8),
        MatValue::Sparse(v) => v.real.len().checked_mul(16),
        MatValue::Cell(values) => return values.iter().any(value_exceeds_level5),
        MatValue::Struct(values) => return values.values().any(value_exceeds_level5),
    };
    bytes.is_none_or(|size| size > LIMIT)
}
