use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use num_complex::Complex32;

use crate::MatError;

pub fn read_be_f32(path: impl AsRef<Path>) -> Result<Vec<f32>, MatError> {
    let mut bytes = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut bytes)?;
    if bytes.len() % 4 != 0 {
        return Err(MatError::RawSize(bytes.len(), 4));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
        .collect())
}

pub fn read_be_complex32(path: impl AsRef<Path>) -> Result<Vec<Complex32>, MatError> {
    let values = read_be_f32(path)?;
    if values.len() % 2 != 0 {
        return Err(MatError::RawSize(values.len() * 4, 8));
    }
    Ok(values
        .chunks_exact(2)
        .map(|pair| Complex32::new(pair[0], pair[1]))
        .collect())
}

pub fn write_be_f32(path: impl AsRef<Path>, values: &[f32]) -> Result<(), MatError> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        writer.write_all(&value.to_be_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn write_be_complex32(path: impl AsRef<Path>, values: &[Complex32]) -> Result<(), MatError> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        writer.write_all(&value.re.to_be_bytes())?;
        writer.write_all(&value.im.to_be_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
