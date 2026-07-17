use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::stats::CandidateData;
use super::SnapPrepError;

pub fn write_patch(
    patch: &Path,
    bounds: [usize; 4],
    noover: [usize; 4],
    data: &CandidateData,
    diff_files: &[PathBuf],
    width: usize,
    length: usize,
) -> Result<usize, SnapPrepError> {
    let [column_start, column_end, row_start, row_end] = bounds;
    let selected = data
        .selected
        .iter()
        .copied()
        .filter(|&(row, column)| {
            (column_start..=column_end).contains(&(column + 1))
                && (row_start..=row_end).contains(&(row + 1))
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Ok(0);
    }
    fs::create_dir_all(patch)?;
    fs::write(
        patch.join("patch.in"),
        bounds
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )?;
    fs::write(
        patch.join("patch_noover.in"),
        noover
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )?;
    write_candidates(patch, &selected, data, width)?;
    write_mean_amplitude(patch, bounds, &data.mean_amplitude, width)?;
    write_phase(patch, &selected, diff_files, width, length)?;
    Ok(selected.len())
}

fn write_candidates(
    patch: &Path,
    selected: &[(usize, usize)],
    data: &CandidateData,
    width: usize,
) -> Result<(), SnapPrepError> {
    let mut ij = BufWriter::new(File::create(patch.join("pscands.1.ij"))?);
    let mut ij_integer = BufWriter::new(File::create(patch.join("pscands.1.ij.int"))?);
    let mut lonlat = BufWriter::new(File::create(patch.join("pscands.1.ll"))?);
    let mut height = BufWriter::new(File::create(patch.join("pscands.1.hgt"))?);
    let mut dispersion = BufWriter::new(File::create(patch.join("pscands.1.da"))?);
    fs::write(patch.join("pscands.1.ij0"), [])?;
    for (index, &(row, column)) in selected.iter().enumerate() {
        writeln!(ij, "{} {row} {column}", index + 1)?;
        ij_integer.write_all(&(column as i32).to_be_bytes())?;
        ij_integer.write_all(&(row as i32).to_be_bytes())?;
        let position = row * width + column;
        lonlat.write_all(&data.lon[position].to_be_bytes())?;
        lonlat.write_all(&data.lat[position].to_be_bytes())?;
        height.write_all(&data.height[position].to_be_bytes())?;
        writeln!(dispersion, "{:.8}", data.dispersion[position])?;
    }
    Ok(())
}

fn write_mean_amplitude(
    patch: &Path,
    bounds: [usize; 4],
    values: &[f32],
    width: usize,
) -> Result<(), SnapPrepError> {
    let [column_start, column_end, row_start, row_end] = bounds;
    let mut output = BufWriter::new(File::create(patch.join("mean_amp.flt"))?);
    for row in row_start - 1..row_end {
        for column in column_start - 1..column_end {
            output.write_all(&values[row * width + column].to_ne_bytes())?;
        }
    }
    Ok(())
}

fn write_phase(
    patch: &Path,
    selected: &[(usize, usize)],
    diff_files: &[PathBuf],
    width: usize,
    length: usize,
) -> Result<(), SnapPrepError> {
    let expected = width
        .checked_mul(length)
        .and_then(|cells| cells.checked_mul(8))
        .ok_or_else(|| SnapPrepError::Invalid("differential raster size overflow".to_owned()))?;
    let mut output =
        BufWriter::with_capacity(1024 * 1024, File::create(patch.join("pscands.1.ph"))?);
    for path in diff_files {
        let actual = fs::metadata(path)?.len() as usize;
        if actual != expected {
            return Err(SnapPrepError::Invalid(format!(
                "unexpected raster size for {}: expected {expected}, found {actual}",
                path.display()
            )));
        }
        let mut bytes = Vec::with_capacity(expected);
        BufReader::with_capacity(1024 * 1024, File::open(path)?).read_to_end(&mut bytes)?;
        for &(row, column) in selected {
            let offset = (row * width + column) * 8;
            output.write_all(&bytes[offset..offset + 8])?;
        }
    }
    output.flush()?;
    Ok(())
}
