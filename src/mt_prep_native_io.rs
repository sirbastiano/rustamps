use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::err;

fn read_text(path: &Path) -> PyResult<String> {
    fs::read_to_string(path).map_err(|e| err(format!("failed to read {}: {e}", path.display())))
}

fn parse_par_int(path: &Path, keys: &[&str]) -> PyResult<Option<usize>> {
    let text = read_text(path)?;
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if keys.iter().any(|candidate| key.trim() == *candidate) {
            let Some(raw) = value.split_whitespace().next() else {
                continue;
            };
            let parsed = raw.parse::<f64>().map_err(|e| {
                PyValueError::new_err(format!(
                    "failed to parse {} in {}: {e}",
                    key.trim(),
                    path.display()
                ))
            })?;
            return Ok(Some(parsed.round() as usize));
        }
    }
    Ok(None)
}

pub(super) fn sorted_files(dir: &Path, suffix: &str) -> PyResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    for entry in
        fs::read_dir(dir).map_err(|e| err(format!("failed to read {}: {e}", dir.display())))?
    {
        let path = entry
            .map_err(|e| err(format!("failed to read {} entry: {e}", dir.display())))?
            .path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|v| v.to_str())
                .is_some_and(|name| name.ends_with(suffix))
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

pub(super) fn dataset_shape(root: &Path) -> PyResult<(usize, usize)> {
    let width_file = root.join("width.txt");
    let len_file = root.join("len.txt");
    if width_file.exists() && len_file.exists() {
        let width = read_text(&width_file)?
            .trim()
            .parse::<usize>()
            .map_err(|e| {
                PyValueError::new_err(format!("failed to parse {}: {e}", width_file.display()))
            })?;
        let length = read_text(&len_file)?.trim().parse::<usize>().map_err(|e| {
            PyValueError::new_err(format!("failed to parse {}: {e}", len_file.display()))
        })?;
        return Ok((width, length));
    }

    let par_files = sorted_files(&root.join("rslc"), ".rslc.par")?;
    let Some(first) = par_files.first() else {
        return Err(err(format!(
            "No rslc/*.rslc.par files found under {}",
            root.display()
        )));
    };
    let width = parse_par_int(first, &["range_samples", "width"])?;
    let length = parse_par_int(first, &["azimuth_lines", "nlines"])?;
    let (Some(width), Some(length)) = (width, length) else {
        return Err(err(format!(
            "Unable to parse raster shape from {}",
            first.display()
        )));
    };
    fs::write(&width_file, format!("{width}\n"))
        .map_err(|e| err(format!("failed to write {}: {e}", width_file.display())))?;
    fs::write(&len_file, format!("{length}\n"))
        .map_err(|e| err(format!("failed to write {}: {e}", len_file.display())))?;
    Ok((width, length))
}

fn pair_dates(name: &str) -> Option<(&str, &str)> {
    let bytes = name.as_bytes();
    if bytes.len() < 17 {
        return None;
    }
    for i in 0..=bytes.len() - 17 {
        if bytes.get(i + 8) != Some(&b'_') {
            continue;
        }
        let master = &name[i..i + 8];
        let slave = &name[i + 9..i + 17];
        if master.bytes().all(|b| b.is_ascii_digit()) && slave.bytes().all(|b| b.is_ascii_digit()) {
            return Some((master, slave));
        }
    }
    None
}

pub(super) fn resolve_master(root: &Path, master_date: Option<&str>) -> PyResult<String> {
    if let Some(master) = master_date.filter(|value| !value.is_empty()) {
        return Ok(master.to_string());
    }
    let root_name = root
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or_default();
    if let Some(pos) = root_name.find("INSAR_") {
        let start = pos + 6;
        if root_name.len() >= start + 8 {
            let candidate = &root_name[start..start + 8];
            if candidate.bytes().all(|b| b.is_ascii_digit()) {
                return Ok(candidate.to_string());
            }
        }
    }
    let mut masters = BTreeSet::new();
    for path in sorted_files(&root.join("diff0"), ".diff")? {
        if let Some(name) = path.file_name().and_then(|v| v.to_str()) {
            if let Some((master, _)) = pair_dates(name) {
                masters.insert(master.to_string());
            }
        }
    }
    if masters.len() == 1 {
        return Ok(masters.into_iter().next().unwrap());
    }
    Err(err(
        "Pass master_date or use a dataset name like INSAR_YYYYMMDD",
    ))
}

pub(super) fn diff_files(root: &Path, master: &str) -> PyResult<Vec<PathBuf>> {
    let mut pairs = Vec::new();
    for path in sorted_files(&root.join("diff0"), ".diff")? {
        if let Some(name) = path.file_name().and_then(|v| v.to_str()) {
            if let Some((found_master, slave)) = pair_dates(name) {
                if found_master == master {
                    pairs.push((slave.to_string(), path));
                }
            }
        }
    }
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    if pairs.is_empty() {
        return Err(err(format!(
            "No diff0/{master}_*.diff files found under {}",
            root.display()
        )));
    }
    Ok(pairs.into_iter().map(|(_, path)| path).collect())
}

pub(super) fn read_file_exact(path: &Path, expected: usize) -> PyResult<Vec<u8>> {
    let actual = fs::metadata(path)
        .map_err(|e| err(format!("failed to stat {}: {e}", path.display())))?
        .len() as usize;
    if actual != expected {
        return Err(err(format!(
            "Unexpected raster size for {}: expected {expected} bytes",
            path.display()
        )));
    }
    let mut data = Vec::with_capacity(expected);
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut data))
        .map_err(|e| err(format!("failed to read {}: {e}", path.display())))?;
    Ok(data)
}

pub(super) fn read_be_f32_raster(path: &Path, cells: usize) -> PyResult<Vec<f32>> {
    let bytes = read_file_exact(path, cells * 4)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
        .collect())
}

pub(super) fn read_be_c64_amplitude(path: &Path, cells: usize) -> PyResult<Vec<f64>> {
    let bytes = read_file_exact(path, cells * 8)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            let re = f32::from_be_bytes(chunk[0..4].try_into().unwrap()) as f64;
            let im = f32::from_be_bytes(chunk[4..8].try_into().unwrap()) as f64;
            (re * re + im * im).sqrt()
        })
        .collect())
}

pub(super) fn ranges(
    size: usize,
    count: usize,
    overlap: usize,
) -> PyResult<Vec<((usize, usize), (usize, usize))>> {
    if count == 0 {
        return Err(err("Patch count must be positive"));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let no_start = (1.0 + (i as f64) * (size as f64) / (count as f64)).floor() as usize;
        let mut no_end =
            (1.0 + ((i + 1) as f64) * (size as f64) / (count as f64)).floor() as usize - 1;
        if i == count - 1 {
            no_end = size;
        }
        out.push((
            (
                no_start.saturating_sub(overlap).max(1),
                (no_end + overlap).min(size),
            ),
            (no_start, no_end),
        ));
    }
    Ok(out)
}
