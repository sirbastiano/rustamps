use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use npyz::WriterBuilder;
use rustamps_io::{MatArray, MatFile, MatValue};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

// Stage 7 contains only 2-D float arrays; keep their names, types and row order.
pub(super) fn write(path: &Path, payload: &MatFile) -> Result<(), Box<dyn std::error::Error>> {
    let mut archive = ZipWriter::new(BufWriter::new(File::create(path)?));
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored)
        .large_file(true);
    for (key, value) in payload {
        archive.start_file(format!("{key}.npy"), options)?;
        let mut entry = BufWriter::new(&mut archive);
        match value {
            MatValue::F32(array) => write_array(&mut entry, array)?,
            MatValue::F64(array) => write_array(&mut entry, array)?,
            _ => return Err(format!("unsupported Stage 7 NumPy array {key}").into()),
        }
        entry.flush()?;
    }
    archive.finish()?.flush()?;
    Ok(())
}

fn write_array<T: npyz::AutoSerialize + Copy>(
    writer: &mut impl Write,
    array: &MatArray<T>,
) -> io::Result<()> {
    let [rows, columns] = array.shape.as_slice() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Stage 7 NumPy arrays must be 2-D",
        ));
    };
    if rows.checked_mul(*columns) != Some(array.values.len()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Stage 7 NumPy array shape mismatch",
        ));
    }
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[*rows as u64, *columns as u64])
        .writer(writer)
        .begin_nd()?;
    writer.extend(array.values.iter().copied())?;
    writer.finish()
}
