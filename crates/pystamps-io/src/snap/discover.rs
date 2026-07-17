use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use super::SnapPrepError;

pub struct SnapDataset {
    pub width: usize,
    pub length: usize,
    pub rslc: Vec<PathBuf>,
    pub diff: Vec<PathBuf>,
    pub lon: PathBuf,
    pub lat: PathBuf,
    pub height: PathBuf,
    pub heading: f64,
    pub wavelength: f64,
}

impl SnapDataset {
    pub fn cells(&self) -> usize {
        self.width * self.length
    }
}

pub fn discover(root: &Path, master_date: Option<&str>) -> Result<SnapDataset, SnapPrepError> {
    let master = resolve_master(root, master_date)?;
    let master_par = root.join("rslc").join(format!("{master}.rslc.par"));
    if !master_par.is_file() {
        return Err(invalid(format!(
            "missing master RSLC parameters {}",
            master_par.display()
        )));
    }
    let (width, length) = dataset_shape(root, &master_par)?;
    let heading = named_scalar(&master_par, "heading")?;
    let frequency = named_scalar(&master_par, "radar_frequency")?;
    if !heading.is_finite() || !frequency.is_finite() || frequency <= 0.0 {
        return Err(invalid(
            "heading must be finite and radar_frequency must be finite and positive",
        ));
    }
    let rslc = sorted_files(&root.join("rslc"), ".rslc")?;
    if rslc.is_empty() {
        return Err(invalid(format!(
            "no rslc/*.rslc files under {}",
            root.display()
        )));
    }
    let mut diff = sorted_files(&root.join("diff0"), ".diff")?
        .into_iter()
        .filter_map(|path| {
            let (found_master, slave) = date_pair(&path)?;
            (found_master == master).then_some((slave, path))
        })
        .collect::<Vec<_>>();
    diff.sort_by(|left, right| left.0.cmp(&right.0));
    if diff.is_empty() {
        return Err(invalid(format!(
            "no diff0/{master}_*.diff files under {}",
            root.display()
        )));
    }
    let lon = root.join("geo").join(format!("{master}.lon"));
    let lat = root.join("geo").join(format!("{master}.lat"));
    let height = root.join("geo/elevation_dem.rdc");
    for path in [&lon, &lat, &height] {
        if !path.is_file() {
            return Err(invalid(format!(
                "missing geocoding raster {}",
                path.display()
            )));
        }
    }
    Ok(SnapDataset {
        width,
        length,
        rslc,
        diff: diff.into_iter().map(|row| row.1).collect(),
        lon,
        lat,
        height,
        heading,
        wavelength: 299_792_458.0 / frequency,
    })
}

pub fn patch_grid(
    width: usize,
    length: usize,
    range_patches: usize,
    azimuth_patches: usize,
    range_overlap: usize,
    azimuth_overlap: usize,
) -> Result<Vec<(usize, [usize; 4], [usize; 4])>, SnapPrepError> {
    let columns = ranges(width, range_patches, range_overlap)?;
    let rows = ranges(length, azimuth_patches, azimuth_overlap)?;
    let mut grid = Vec::with_capacity(columns.len() * rows.len());
    let mut index = 1;
    for (column, column_noover) in columns {
        for &(row, row_noover) in &rows {
            grid.push((
                index,
                [column.0, column.1, row.0, row.1],
                [column_noover.0, column_noover.1, row_noover.0, row_noover.1],
            ));
            index += 1;
        }
    }
    Ok(grid)
}

fn ranges(
    size: usize,
    count: usize,
    overlap: usize,
) -> Result<Vec<((usize, usize), (usize, usize))>, SnapPrepError> {
    if size == 0 || count == 0 || count > size {
        return Err(invalid(
            "raster dimensions and patch counts must be positive",
        ));
    }
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let no_start = (1.0 + index as f64 * size as f64 / count as f64).floor() as usize;
        let mut no_end =
            (1.0 + (index + 1) as f64 * size as f64 / count as f64).floor() as usize - 1;
        if index + 1 == count {
            no_end = size;
        }
        output.push((
            (
                no_start.saturating_sub(overlap).max(1),
                no_end.saturating_add(overlap).min(size),
            ),
            (no_start, no_end),
        ));
    }
    Ok(output)
}

fn dataset_shape(root: &Path, par: &Path) -> Result<(usize, usize), SnapPrepError> {
    let width_file = root.join("width.txt");
    let length_file = root.join("len.txt");
    if width_file.is_file() && length_file.is_file() {
        return Ok((parse_usize(&width_file)?, parse_usize(&length_file)?));
    }
    let width = named_integer(par, &["range_samples", "width"])?;
    let length = named_integer(par, &["azimuth_lines", "nlines"])?;
    fs::write(width_file, format!("{width}\n"))?;
    fs::write(length_file, format!("{length}\n"))?;
    Ok((width, length))
}

fn resolve_master(root: &Path, requested: Option<&str>) -> Result<String, SnapPrepError> {
    if let Some(value) = requested.filter(|value| !value.is_empty()) {
        return Ok(value.to_owned());
    }
    let name = root
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if let Some(start) = name.find("INSAR_").map(|value| value + 6) {
        if name
            .get(start..start + 8)
            .is_some_and(|value| value.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Ok(name[start..start + 8].to_owned());
        }
    }
    let masters = sorted_files(&root.join("diff0"), ".diff")?
        .iter()
        .filter_map(|path| date_pair(path).map(|pair| pair.0))
        .collect::<BTreeSet<_>>();
    if masters.len() == 1 {
        return Ok(masters.into_iter().next().unwrap());
    }
    Err(invalid(
        "pass master_date or use a root name containing INSAR_YYYYMMDD",
    ))
}

fn sorted_files(directory: &Path, suffix: &str) -> Result<Vec<PathBuf>, SnapPrepError> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            path.file_name()
                .and_then(|v| v.to_str())
                .is_some_and(|v| v.ends_with(suffix))
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn date_pair(path: &Path) -> Option<(String, String)> {
    let name = path.file_name()?.to_str()?;
    name.as_bytes().windows(17).find_map(|part| {
        if part[8] != b'_'
            || !part[..8].iter().all(u8::is_ascii_digit)
            || !part[9..].iter().all(u8::is_ascii_digit)
        {
            return None;
        }
        Some((
            String::from_utf8(part[..8].to_vec()).ok()?,
            String::from_utf8(part[9..].to_vec()).ok()?,
        ))
    })
}

fn named_integer(path: &Path, keys: &[&str]) -> Result<usize, SnapPrepError> {
    let text = fs::read_to_string(path)?;
    for line in text.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if keys.contains(&name.trim()) {
            let number = value
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok());
            if let Some(number) = number {
                return Ok(number.round() as usize);
            }
        }
    }
    Err(invalid(format!(
        "unable to parse dimensions from {}",
        path.display()
    )))
}

fn named_scalar(path: &Path, key: &str) -> Result<f64, SnapPrepError> {
    let text = fs::read_to_string(path)?;
    for line in text.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim() == key {
            return value
                .split_whitespace()
                .next()
                .ok_or_else(|| invalid(format!("missing value for {key}")))?
                .parse::<f64>()
                .map_err(|error| invalid(format!("invalid {key}: {error}")));
        }
    }
    Err(invalid(format!("missing {key} in {}", path.display())))
}

fn parse_usize(path: &Path) -> Result<usize, SnapPrepError> {
    fs::read_to_string(path)?
        .trim()
        .parse()
        .map_err(|error| invalid(format!("{}: {error}", path.display())))
}

fn invalid(message: impl Into<String>) -> SnapPrepError {
    SnapPrepError::Invalid(message.into())
}
