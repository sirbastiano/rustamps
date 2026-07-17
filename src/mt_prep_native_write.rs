use pyo3::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::err;
use super::mt_prep_native_io::read_file_exact;

fn write_be_f32(writer: &mut File, value: f32) -> PyResult<()> {
    writer
        .write_all(&value.to_be_bytes())
        .map_err(|e| err(format!("failed to write float: {e}")))
}

fn write_native_f32(writer: &mut File, value: f32) -> PyResult<()> {
    writer
        .write_all(&value.to_ne_bytes())
        .map_err(|e| err(format!("failed to write float: {e}")))
}

pub(super) fn write_patch(
    patch: &Path,
    bounds: (usize, usize, usize, usize),
    noover: (usize, usize, usize, usize),
    selected: &[(usize, usize)],
    rasters: (&[f32], &[f32], &[f32], &[f32], &[f32]),
    diff_files: &[PathBuf],
    cells: usize,
    width: usize,
) -> PyResult<usize> {
    let (c0, c1, r0, r1) = bounds;
    let rows_cols: Vec<(usize, usize)> = selected
        .iter()
        .copied()
        .filter(|(row, col)| col + 1 >= c0 && col + 1 <= c1 && row + 1 >= r0 && row + 1 <= r1)
        .collect();
    if rows_cols.is_empty() {
        return Ok(0);
    }

    fs::create_dir_all(patch)
        .map_err(|e| err(format!("failed to create {}: {e}", patch.display())))?;
    fs::write(patch.join("patch.in"), format!("{c0}\n{c1}\n{r0}\n{r1}\n"))
        .map_err(|e| err(format!("failed to write patch.in: {e}")))?;
    fs::write(
        patch.join("patch_noover.in"),
        format!("{}\n{}\n{}\n{}\n", noover.0, noover.1, noover.2, noover.3),
    )
    .map_err(|e| err(format!("failed to write patch_noover.in: {e}")))?;

    let mut ij = File::create(patch.join("pscands.1.ij"))
        .map_err(|e| err(format!("failed to write ij: {e}")))?;
    let mut ij_int = File::create(patch.join("pscands.1.ij.int"))
        .map_err(|e| err(format!("failed to write ij.int: {e}")))?;
    fs::write(patch.join("pscands.1.ij0"), "")
        .map_err(|e| err(format!("failed to write ij0: {e}")))?;
    let (lon, lat, hgt, da, mean_amp) = rasters;
    let mut ll = File::create(patch.join("pscands.1.ll"))
        .map_err(|e| err(format!("failed to write ll: {e}")))?;
    let mut hgt_file = File::create(patch.join("pscands.1.hgt"))
        .map_err(|e| err(format!("failed to write hgt: {e}")))?;
    let mut da_text = String::new();

    for (idx, (row, col)) in rows_cols.iter().enumerate() {
        writeln!(ij, "{} {} {}", idx + 1, row, col)
            .map_err(|e| err(format!("failed to write ij: {e}")))?;
        ij_int
            .write_all(&(*col as i32).to_be_bytes())
            .map_err(|e| err(format!("failed to write ij.int: {e}")))?;
        ij_int
            .write_all(&(*row as i32).to_be_bytes())
            .map_err(|e| err(format!("failed to write ij.int: {e}")))?;
        let pos = row * width + col;
        write_be_f32(&mut ll, lon[pos])?;
        write_be_f32(&mut ll, lat[pos])?;
        write_be_f32(&mut hgt_file, hgt[pos])?;
        da_text.push_str(&format!("{:.8}\n", da[pos]));
    }
    fs::write(patch.join("pscands.1.da"), da_text)
        .map_err(|e| err(format!("failed to write da: {e}")))?;

    let mut amp_file = File::create(patch.join("mean_amp.flt"))
        .map_err(|e| err(format!("failed to write mean_amp: {e}")))?;
    for row in (r0 - 1)..r1 {
        for col in (c0 - 1)..c1 {
            write_native_f32(&mut amp_file, mean_amp[row * width + col])?;
        }
    }

    let mut ph = File::create(patch.join("pscands.1.ph"))
        .map_err(|e| err(format!("failed to write ph: {e}")))?;
    for diff in diff_files {
        let bytes = read_file_exact(diff, cells * 8)?;
        for (row, col) in &rows_cols {
            let pos = (row * width + col) * 8;
            ph.write_all(&bytes[pos..pos + 8])
                .map_err(|e| err(format!("failed to write ph: {e}")))?;
        }
    }
    Ok(rows_cols.len())
}
