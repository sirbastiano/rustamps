use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use pystamps_core::stages::stage1::Matrix;
use pystamps_io::atomic_write;

pub struct Metadata {
    pub days: Vec<i32>,
    pub master: i32,
    pub bperp: Vec<f64>,
    pub bperp_mat: Option<Matrix<f32>>,
    pub heading_deg: Option<f64>,
    pub mean_range: f64,
    pub mean_incidence: f64,
    pub look_angle: Option<Vec<f64>>,
    pub wavelength: Option<f64>,
}

pub fn resolve(patch: &Path, ij: &[f64], n_ps: usize) -> Result<Metadata, String> {
    let snap_root = patch
        .parent()
        .is_some_and(|root| root.join("diff0").is_dir() && root.join("rslc").is_dir());
    if snap_root {
        return synthesize(patch, ij, n_ps);
    }
    super::legacy_guard::reject_unsupported_spatial_inputs(patch)?;
    let explicit = (
        ancestor_file(patch, "day.1.in"),
        ancestor_file(patch, "master_day.1.in"),
        ancestor_file(patch, "bperp.1.in"),
    );
    if let (Some(day), Some(master), Some(bperp)) = explicit {
        return Ok(Metadata {
            days: integers(&day)?,
            master: one_integer(&master)?,
            bperp: numbers(&bperp)?,
            bperp_mat: None,
            heading_deg: None,
            mean_range: 830_000.0,
            mean_incidence: 23_f64.to_radians(),
            look_angle: None,
            wavelength: None,
        });
    }
    synthesize(patch, ij, n_ps)
}

fn synthesize(patch: &Path, ij: &[f64], n_ps: usize) -> Result<Metadata, String> {
    let root = patch
        .parent()
        .ok_or_else(|| "patch has no dataset parent".to_owned())?;
    let mut records = Vec::new();
    for entry in fs::read_dir(root.join("diff0")).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().and_then(|v| v.to_str()) != Some("base") {
            continue;
        }
        if let Some((master, slave)) = date_pair(&path) {
            records.push((master, slave, path));
        }
    }
    records.sort_by(|left, right| left.2.cmp(&right.2));
    if records.is_empty() {
        return Err(format!(
            "no parseable diff0/*.base metadata under {}",
            root.display()
        ));
    }
    let masters = records.iter().map(|row| row.0).collect::<BTreeSet<_>>();
    if masters.len() != 1 {
        return Err("SNAP metadata synthesis requires a single-master stack".to_owned());
    }
    let master = *masters.first().unwrap();
    let par = resolve_rslc_par(root, master)?;
    let geometry = Geometry::read(&par)?;
    let mut columns = Vec::with_capacity(records.len());
    let mut means = Vec::with_capacity(records.len());
    for (_, _, path) in &records {
        let column = baseline_column(path, &geometry, ij, n_ps)?;
        means.push(column.iter().map(|&v| f64::from(v)).sum::<f64>() / n_ps as f64);
        columns.push(column);
    }
    let mut values = Vec::with_capacity(n_ps * columns.len());
    for row in 0..n_ps {
        for column in &columns {
            values.push(column[row]);
        }
    }
    let days = records.iter().map(|row| row.1).collect::<Vec<_>>();
    write_metadata_inputs(patch, &days, master, &means)?;
    let look_angle = look_angles(&geometry, ij, n_ps);
    let mean_incidence = look_angle.iter().sum::<f64>() / n_ps as f64;
    Ok(Metadata {
        days,
        master,
        bperp: means,
        bperp_mat: Some(Matrix::new(n_ps, columns.len(), values).map_err(|e| e.to_string())?),
        heading_deg: Some(geometry.heading),
        mean_range: geometry.center_range,
        mean_incidence,
        look_angle: Some(look_angle),
        wavelength: Some(geometry.wavelength),
    })
}

struct Geometry {
    range_spacing: f64,
    near_range: f64,
    center_range: f64,
    sensor_radius: f64,
    earth_radius: f64,
    azimuth_lines: f64,
    prf: f64,
    heading: f64,
    wavelength: f64,
}

impl Geometry {
    fn read(path: &Path) -> Result<Self, String> {
        let frequency = named_scalar(path, "radar_frequency")?;
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(format!("invalid radar_frequency in {}", path.display()));
        }
        Ok(Self {
            range_spacing: named_scalar(path, "range_pixel_spacing")?,
            near_range: named_scalar(path, "near_range_slc")?,
            center_range: named_scalar(path, "center_range_slc").unwrap_or(830_000.0),
            sensor_radius: named_scalar(path, "sar_to_earth_center")?,
            earth_radius: named_scalar(path, "earth_radius_below_sensor")?,
            azimuth_lines: named_scalar(path, "azimuth_lines")?,
            prf: named_scalar(path, "prf")?,
            heading: named_scalar(path, "heading")?,
            wavelength: 299_792_458.0 / frequency,
        })
    }
}

fn baseline_column(
    path: &Path,
    geo: &Geometry,
    ij: &[f64],
    rows: usize,
) -> Result<Vec<f32>, String> {
    let baseline = named_vector(path, "initial_baseline(TCN)", 3)?;
    let rate = named_vector(path, "initial_baseline_rate", 3)?;
    if geo.prf == 0.0 {
        return Err(format!("zero PRF in {}", path.display()));
    }
    let mean_azimuth = geo.azimuth_lines / 2.0 - 0.5;
    Ok((0..rows)
        .map(|row| {
            let azimuth = ij[row * 3 + 1];
            let range = geo.near_range + ij[row * 3 + 2] * geo.range_spacing;
            let look = ((geo.sensor_radius.powi(2) + range.powi(2) - geo.earth_radius.powi(2))
                / (2.0 * geo.sensor_radius * range))
                .clamp(-1.0, 1.0)
                .acos();
            let bc = baseline[1] + rate[1] * (azimuth - mean_azimuth) / geo.prf;
            let bn = baseline[2] + rate[2] * (azimuth - mean_azimuth) / geo.prf;
            (bc * look.cos() - bn * look.sin()) as f32
        })
        .collect())
}

fn look_angles(geo: &Geometry, ij: &[f64], rows: usize) -> Vec<f64> {
    (0..rows)
        .map(|row| {
            let range = geo.near_range + ij[row * 3 + 2] * geo.range_spacing;
            ((geo.sensor_radius.powi(2) - geo.earth_radius.powi(2) - range.powi(2))
                / (2.0 * geo.earth_radius * range))
                .clamp(-1.0, 1.0)
                .acos()
        })
        .collect()
}

fn resolve_rslc_par(root: &Path, master: i32) -> Result<PathBuf, String> {
    let preferred = root.join("rslc").join(format!("{master}.rslc.par"));
    if preferred.exists() {
        return Ok(preferred);
    }
    Err(format!(
        "missing master RSLC parameters: {}",
        preferred.display()
    ))
}

fn date_pair(path: &Path) -> Option<(i32, i32)> {
    let name = path.file_name()?.to_str()?;
    name.as_bytes().windows(17).find_map(|window| {
        (window[8] == b'_'
            && window[..8].iter().all(u8::is_ascii_digit)
            && window[9..].iter().all(u8::is_ascii_digit))
        .then(|| {
            let text = std::str::from_utf8(window).ok()?;
            Some((text[..8].parse().ok()?, text[9..].parse().ok()?))
        })
        .flatten()
    })
}

fn named_scalar(path: &Path, key: &str) -> Result<f64, String> {
    Ok(named_vector(path, key, 1)?[0])
}

fn named_vector(path: &Path, key: &str, count: usize) -> Result<Vec<f64>, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    for line in text.lines() {
        let Some((name, tail)) = line.split_once(':') else {
            continue;
        };
        if name.trim() != key {
            continue;
        }
        let values = tail
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter_map(|token| token.parse::<f64>().ok())
            .take(count)
            .collect::<Vec<_>>();
        if values.len() == count {
            return Ok(values);
        }
    }
    Err(format!("unable to parse {key} from {}", path.display()))
}

fn ancestor_file(patch: &Path, name: &str) -> Option<PathBuf> {
    [
        patch.to_path_buf(),
        patch.parent()?.to_path_buf(),
        patch.parent()?.parent()?.to_path_buf(),
    ]
    .into_iter()
    .map(|root| root.join(name))
    .find(|path| path.exists())
}

fn numbers(path: &Path) -> Result<Vec<f64>, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    text.split_whitespace()
        .map(|v| v.parse::<f64>().map_err(|e| e.to_string()))
        .collect()
}

fn integers(path: &Path) -> Result<Vec<i32>, String> {
    numbers(path)?
        .into_iter()
        .map(|v| i32::try_from(v.round() as i64).map_err(|e| e.to_string()))
        .collect()
}

fn one_integer(path: &Path) -> Result<i32, String> {
    integers(path)?
        .first()
        .copied()
        .ok_or_else(|| format!("{} is empty", path.display()))
}

fn write_metadata_inputs(
    patch: &Path,
    days: &[i32],
    master: i32,
    bperp: &[f64],
) -> Result<(), String> {
    let day = days
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let baseline = bperp
        .iter()
        .map(|v| format!("{v:.12}"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    atomic_write(patch.join("day.1.in"), day.as_bytes()).map_err(|e| e.to_string())?;
    atomic_write(
        patch.join("master_day.1.in"),
        format!("{master}\n").as_bytes(),
    )
    .map_err(|e| e.to_string())?;
    atomic_write(patch.join("bperp.1.in"), baseline.as_bytes()).map_err(|e| e.to_string())
}
