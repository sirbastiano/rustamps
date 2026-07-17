use std::fs;
use std::path::{Path, PathBuf};

use pystamps_core::stages::stage5::{
    merge_patches_with_heading, NoOverlapBounds, PromotedPatch, Stage5Row,
};
use pystamps_io::{discover_dataset, read_mat, MatFile};

use crate::{PipelineError, RunConfig};

use super::super::failure;
use super::super::mat::{complex32, numeric_f32, numeric_f64, scalar_f64, shape};
use super::super::params::Params;

pub fn run(root: &Path, _config: &RunConfig) -> Result<String, PipelineError> {
    let params = Params::load(root).map_err(|error| failure(5, error))?;
    if params
        .flag("small_baseline_flag", false)
        .map_err(|error| failure(5, error))?
    {
        return Err(failure(
            5,
            "small-baseline Stage 5 merge is not yet supported",
        ));
    }
    let resample = params
        .scalar("merge_resample_size", 0.0)
        .map_err(|error| failure(5, error))?;
    let heading = params
        .scalar("heading", f64::NAN)
        .map_err(|error| failure(5, error))?;
    if !heading.is_finite() {
        return Err(failure(5, "heading must be present and finite"));
    }
    let layout = discover_dataset(root).map_err(|error| failure(5, error))?;
    let ownership = patch_ownership(&layout.patches).map_err(|error| failure(5, error))?;
    let mut patches = Vec::with_capacity(layout.patches.len());
    let mut base = None;
    let mut n_ifg = None;
    let mut master_ix = None;
    for (path, no_overlap) in layout.patches.iter().zip(ownership) {
        let loaded = load_patch(path, no_overlap).map_err(|error| failure(5, error))?;
        if n_ifg.is_some_and(|value| value != loaded.n_ifg)
            || master_ix.is_some_and(|value| value != loaded.master_ix)
        {
            return Err(failure(5, "patch interferogram metadata is inconsistent"));
        }
        n_ifg = Some(loaded.n_ifg);
        master_ix = Some(loaded.master_ix);
        if base.is_none() {
            base = Some(loaded.ps);
        }
        patches.push(loaded.patch);
    }
    let merged = merge_patches_with_heading(&patches, resample, Some(heading))
        .map_err(|error| failure(5, error))?;
    let count = merged.rows.len();
    super::write::merged(
        root,
        &base.ok_or_else(|| failure(5, "no patch PS data available"))?,
        merged,
        master_ix.unwrap_or(1),
        n_ifg.unwrap_or(0),
    )
    .map_err(|error| failure(5, error))?;
    Ok(format!(
        "Merged {} patches into {count} PS records",
        layout.patches.len()
    ))
}

struct LoadedPatch {
    ps: MatFile,
    patch: PromotedPatch,
    n_ifg: usize,
    master_ix: usize,
}

fn load_patch(path: &Path, no_overlap: Option<NoOverlapBounds>) -> Result<LoadedPatch, String> {
    let ps = required(path, "ps2.mat")?;
    let phase_file = required(path, "ph2.mat")?;
    let pm = required(path, "pm2.mat")?;
    let n_ps = integer_scalar(&ps, "n_ps")?;
    let n_ifg = integer_scalar(&ps, "n_ifg")?;
    let master_ix = integer_scalar(&ps, "master_ix")?;
    if n_ps == 0 || !(1..=n_ifg).contains(&master_ix) {
        return Err(format!("{}/ps2.mat has invalid dimensions", path.display()));
    }
    require_shape(&ps, "ij", n_ps, 3)?;
    require_shape(&ps, "lonlat", n_ps, 2)?;
    require_shape(&phase_file, "ph", n_ps, n_ifg)?;
    require_shape(&pm, "ph_patch", n_ps, n_ifg - 1)?;
    require_shape(&pm, "ph_res", n_ps, n_ifg - 1)?;
    let ij = numeric_f64(&ps, "ij")?;
    let lonlat = numeric_f64(&ps, "lonlat")?;
    let phase = complex32(&phase_file, "ph")?;
    let k = numeric_f64(&pm, "K_ps")?;
    let c = numeric_f64(&pm, "C_ps")?;
    let coherence = numeric_f64(&pm, "coh_ps")?;
    let patch_phase = complex32(&pm, "ph_patch")?;
    let residual = numeric_f32(&pm, "ph_res")?;
    for (name, values) in [("K_ps", &k), ("C_ps", &c), ("coh_ps", &coherence)] {
        if values.len() != n_ps {
            return Err(format!("{}/pm2.{name} row mismatch", path.display()));
        }
    }
    let baseline = optional_f32(path, "bp2.mat", "bperp_mat", n_ps * (n_ifg - 1))?;
    let height = optional_f32(path, "hgt2.mat", "hgt", n_ps)?;
    let look_angle = optional_f64(path, "la2.mat", "la", n_ps)?;
    let dispersion = optional_f64(path, "da2.mat", "D_A", n_ps)?;
    let rows = (0..n_ps)
        .map(|row| Stage5Row {
            ij: [ij[row * 3], ij[row * 3 + 1], ij[row * 3 + 2]],
            lonlat: [lonlat[row * 2], lonlat[row * 2 + 1]],
            phase: phase[row * n_ifg..(row + 1) * n_ifg].to_vec(),
            k_ps: k[row],
            c_ps: c[row],
            coherence: coherence[row],
            phase_patch: patch_phase[row * (n_ifg - 1)..(row + 1) * (n_ifg - 1)].to_vec(),
            phase_residual: residual[row * (n_ifg - 1)..(row + 1) * (n_ifg - 1)].to_vec(),
            bperp: baseline
                .as_ref()
                .map(|values| values[row * (n_ifg - 1)..(row + 1) * (n_ifg - 1)].to_vec()),
            height: height.as_ref().map(|values| values[row]),
            look_angle: look_angle.as_ref().map(|values| values[row]),
            amplitude_dispersion: dispersion.as_ref().map(|values| values[row]),
        })
        .collect();
    Ok(LoadedPatch {
        ps,
        n_ifg,
        master_ix,
        patch: PromotedPatch {
            name: path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("PATCH")
                .to_owned(),
            no_overlap,
            rows,
        },
    })
}

fn patch_ownership(paths: &[PathBuf]) -> Result<Vec<Option<NoOverlapBounds>>, String> {
    let required = paths.len() > 1;
    paths
        .iter()
        .map(|path| {
            let bounds = no_overlap(path)?;
            if required && bounds.is_none() {
                Err(format!(
                    "missing required Stage 5 patch ownership artifact {}",
                    path.join("patch_noover.in").display()
                ))
            } else {
                Ok(bounds)
            }
        })
        .collect()
}

fn no_overlap(path: &Path) -> Result<Option<NoOverlapBounds>, String> {
    let target = path.join("patch_noover.in");
    if !target.is_file() {
        return Ok(None);
    }
    let values = fs::read_to_string(target)
        .map_err(|error| error.to_string())?
        .split_whitespace()
        .map(|value| value.parse::<i64>().map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() < 4 {
        return Err("patch_noover.in requires four integers".to_owned());
    }
    Ok(Some(NoOverlapBounds {
        row_min: values[0],
        row_max: values[1],
        column_min: values[2],
        column_max: values[3],
    }))
}

fn required(path: &Path, name: &str) -> Result<MatFile, String> {
    let target = path.join(name);
    if !target.is_file() {
        return Err(format!(
            "missing patch Stage 5 artifact {}",
            target.display()
        ));
    }
    read_mat(target).map_err(|error| error.to_string())
}

fn integer_scalar(file: &MatFile, key: &str) -> Result<usize, String> {
    let value = scalar_f64(file, key)?;
    if value < 0.0 || !value.is_finite() || value.fract() != 0.0 {
        Err(format!("{key} is not a non-negative integer"))
    } else {
        Ok(value as usize)
    }
}

fn require_shape(file: &MatFile, key: &str, rows: usize, cols: usize) -> Result<(), String> {
    let dimensions = shape(file, key)?;
    (dimensions == [rows, cols])
        .then_some(())
        .ok_or_else(|| format!("{key} has shape {dimensions:?}; expected [{rows}, {cols}]"))
}

fn optional_f32(
    path: &Path,
    name: &str,
    key: &str,
    expected: usize,
) -> Result<Option<Vec<f32>>, String> {
    if !path.join(name).is_file() {
        return Ok(None);
    }
    let values = numeric_f32(&required(path, name)?, key)?;
    (values.len() == expected)
        .then_some(Some(values))
        .ok_or_else(|| format!("{name}.{key} size mismatch"))
}

fn optional_f64(
    path: &Path,
    name: &str,
    key: &str,
    expected: usize,
) -> Result<Option<Vec<f64>>, String> {
    if !path.join(name).is_file() {
        return Ok(None);
    }
    let values = numeric_f64(&required(path, name)?, key)?;
    (values.len() == expected)
        .then_some(Some(values))
        .ok_or_else(|| format!("{name}.{key} size mismatch"))
}
