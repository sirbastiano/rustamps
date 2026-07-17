use std::fs;
use std::path::Path;

use num_complex::Complex32;
use pystamps_io::{read_be_complex32, read_be_f32};

pub fn text_matrix(path: &Path, columns: usize) -> Result<Vec<f64>, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let values = text
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|e| format!("invalid number in {}: {e}", path.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if columns == 0 || values.len() % columns != 0 {
        return Err(format!(
            "{} contains {} values, not a multiple of {columns}",
            path.display(),
            values.len()
        ));
    }
    Ok(values)
}

pub fn be_f32_matrix(path: &Path, rows: usize, columns: usize) -> Result<Vec<f32>, String> {
    let values = read_be_f32(path).map_err(|e| e.to_string())?;
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| "raw matrix size overflow".to_owned())?;
    if values.len() != expected {
        return Err(format!(
            "{} has {} f32 values; expected {rows}x{columns}",
            path.display(),
            values.len()
        ));
    }
    Ok(values)
}

pub fn phase_matrix(path: &Path, rows: usize, columns: usize) -> Result<Vec<Complex32>, String> {
    let source = read_be_complex32(path).map_err(|e| e.to_string())?;
    if source.len() != rows.saturating_mul(columns) {
        return Err(format!(
            "{} has {} complex values; expected {rows}x{columns}",
            path.display(),
            source.len()
        ));
    }
    let mut values = vec![Complex32::new(0.0, 0.0); source.len()];
    for column in 0..columns {
        for row in 0..rows {
            values[row * columns + column] = source[column * rows + row];
        }
    }
    Ok(values)
}

pub fn optional_text_vector(path: &Path, expected: usize) -> Result<Option<Vec<f64>>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let values = text_matrix(path, 1)?;
    require_len(path, values.len(), expected)?;
    Ok(Some(values))
}

pub fn optional_be_f32_vector(path: &Path, expected: usize) -> Result<Option<Vec<f32>>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let values = read_be_f32(path).map_err(|e| e.to_string())?;
    require_len(path, values.len(), expected)?;
    Ok(Some(values))
}

fn require_len(path: &Path, actual: usize, expected: usize) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{} has {actual} values; expected {expected}",
            path.display()
        ))
    }
}
