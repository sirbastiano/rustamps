use std::collections::BTreeMap;
use std::path::Path;

use rustamps_core::stages::stage1::Stage1Output;
use rustamps_io::{write_mat, MatFile, StageTransaction};

use super::super::mat::{complex32_array, f32_array, f64_array, scalar};

pub fn write(
    patch: &Path,
    output: Stage1Output,
    mean_range: f64,
    mean_incidence: f64,
    look_angle: Option<Vec<f64>>,
) -> Result<(), String> {
    let transaction = StageTransaction::begin(patch, "stage1").map_err(|e| e.to_string())?;
    let n_ps = output.ij.rows;
    let n_ifg = output.phase.cols;
    let look_angle = look_angle
        .map(|values| {
            output
                .sort_ix
                .iter()
                .map(|&one_based| {
                    values
                        .get(one_based - 1)
                        .copied()
                        .ok_or_else(|| "look-angle source does not match Stage 1 rows".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let has_da = output.amplitude_dispersion.is_some();
    let has_height = output.height.is_some();
    let has_look_angle = look_angle.is_some();
    let mut ps = MatFile::new();
    ps.insert("ij".to_owned(), f64_array(vec![n_ps, 3], output.ij.values));
    ps.insert(
        "lonlat".to_owned(),
        f64_array(vec![n_ps, 2], output.lonlat.values),
    );
    ps.insert("xy".to_owned(), f32_array(vec![n_ps, 3], output.xy.values));
    ps.insert("bperp".to_owned(), f32_array(vec![n_ifg, 1], output.bperp));
    ps.insert("day".to_owned(), f64_array(vec![n_ifg, 1], output.day));
    ps.insert("master_day".to_owned(), scalar(output.master_day));
    ps.insert("master_ix".to_owned(), scalar(output.master_ix as f64));
    ps.insert("n_ifg".to_owned(), scalar(n_ifg as f64));
    ps.insert("n_image".to_owned(), scalar(n_ifg as f64));
    ps.insert("n_ps".to_owned(), scalar(n_ps as f64));
    ps.insert(
        "sort_ix".to_owned(),
        f64_array(
            vec![n_ps, 1],
            output
                .sort_ix
                .into_iter()
                .map(|value| value as f64)
                .collect(),
        ),
    );
    ps.insert("ll0".to_owned(), f64_array(vec![1, 2], output.ll0.to_vec()));
    ps.insert("mean_range".to_owned(), scalar(mean_range));
    ps.insert("mean_incidence".to_owned(), scalar(mean_incidence));

    let mut ph = BTreeMap::new();
    ph.insert(
        "ph".to_owned(),
        complex32_array(vec![n_ps, n_ifg], output.phase.values),
    );
    write_mat(transaction.path("ph1.mat"), &ph).map_err(|e| e.to_string())?;

    let mut bp = BTreeMap::new();
    bp.insert(
        "bperp_mat".to_owned(),
        f32_array(
            vec![output.bperp_mat.rows, output.bperp_mat.cols],
            output.bperp_mat.values,
        ),
    );
    write_mat(transaction.path("bp1.mat"), &bp).map_err(|e| e.to_string())?;

    let mut files = vec!["ph1.mat", "bp1.mat", "psver.mat"];
    let mut version = BTreeMap::new();
    version.insert("psver".to_owned(), scalar(1.0));
    write_mat(transaction.path("psver.mat"), &version).map_err(|e| e.to_string())?;
    if let Some(values) = output.amplitude_dispersion {
        let mut payload = BTreeMap::new();
        payload.insert("D_A".to_owned(), f64_array(vec![n_ps, 1], values));
        write_mat(transaction.path("da1.mat"), &payload).map_err(|e| e.to_string())?;
        files.push("da1.mat");
    }
    if let Some(values) = output.height {
        let mut payload = BTreeMap::new();
        payload.insert("hgt".to_owned(), f32_array(vec![n_ps, 1], values));
        write_mat(transaction.path("hgt1.mat"), &payload).map_err(|e| e.to_string())?;
        files.push("hgt1.mat");
    }
    if let Some(values) = look_angle {
        let mut payload = BTreeMap::new();
        payload.insert("la".to_owned(), f64_array(vec![n_ps, 1], values));
        write_mat(transaction.path("la1.mat"), &payload).map_err(|e| e.to_string())?;
        files.push("la1.mat");
    }
    write_mat(transaction.path("ps1.mat"), &ps).map_err(|e| e.to_string())?;
    files.push("ps1.mat");
    let mut removals = vec!["inc1.mat"];
    if !has_da {
        removals.push("da1.mat");
    }
    if !has_height {
        removals.push("hgt1.mat");
    }
    if !has_look_angle {
        removals.push("la1.mat");
    }
    transaction
        .commit_with_removals(&files, "ps1.mat", &removals)
        .map_err(|e| e.to_string())
}
