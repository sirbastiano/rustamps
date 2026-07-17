use std::path::Path;

use rustamps_core::stages::stage1::Matrix;
use rustamps_core::stages::stage5::{rc2_correction, Stage5Merged};
use rustamps_io::{write_mat, MatFile, StageTransaction};

use super::super::mat::{complex32_array, f32_array, f64_array, numeric_f32, numeric_f64, scalar};
use super::patch::PatchProduct;

pub fn patch(path: &Path, product: PatchProduct) -> Result<(), String> {
    let transaction =
        StageTransaction::begin(path, "stage5-patch").map_err(|error| error.to_string())?;
    let n_ps = product.rows.rows.len();
    let n_ifg = product.n_ifg;
    let n_patch_ifg = n_ifg - 1;
    let rows = &product.rows.rows;
    let phase = Matrix::new(
        n_ps,
        n_ifg,
        rows.iter()
            .flat_map(|row| row.phase.iter().copied())
            .collect(),
    )
    .map_err(|error| error.to_string())?;
    let patch_phase = Matrix::new(
        n_ps,
        n_patch_ifg,
        rows.iter()
            .flat_map(|row| row.phase_patch.iter().copied())
            .collect(),
    )
    .map_err(|error| error.to_string())?;
    let baseline = baseline_matrix(rows, &product.ps, n_ifg, product.master_ix)?;
    let k = rows.iter().map(|row| row.k_ps).collect::<Vec<_>>();
    let c = rows.iter().map(|row| row.c_ps).collect::<Vec<_>>();
    let corrected = rc2_correction(
        &phase,
        &patch_phase,
        &baseline,
        &k,
        &c,
        false,
        product.master_ix,
    )
    .map_err(|error| error.to_string())?;

    let mut ps2 = base_ps(&product.ps, n_ps, n_ifg)?;
    ps2.insert(
        "ij".to_owned(),
        f64_array(vec![n_ps, 3], rows.iter().flat_map(|row| row.ij).collect()),
    );
    ps2.insert(
        "lonlat".to_owned(),
        f64_array(
            vec![n_ps, 2],
            rows.iter().flat_map(|row| row.lonlat).collect(),
        ),
    );
    ps2.insert("xy".to_owned(), f32_array(vec![n_ps, 3], product.xy));

    let mut ph2 = MatFile::new();
    ph2.insert(
        "ph".to_owned(),
        complex32_array(vec![n_ps, n_ifg], phase.values),
    );
    let mut pm2 = MatFile::new();
    pm2.insert("K_ps".to_owned(), f64_array(vec![n_ps, 1], k));
    pm2.insert("C_ps".to_owned(), f64_array(vec![n_ps, 1], c));
    pm2.insert(
        "coh_ps".to_owned(),
        f64_array(
            vec![n_ps, 1],
            rows.iter().map(|row| row.coherence).collect(),
        ),
    );
    pm2.insert(
        "ph_patch".to_owned(),
        complex32_array(vec![n_ps, n_patch_ifg], patch_phase.values),
    );
    pm2.insert(
        "ph_res".to_owned(),
        f32_array(
            vec![n_ps, n_patch_ifg],
            rows.iter()
                .flat_map(|row| row.phase_residual.iter().copied())
                .collect(),
        ),
    );
    let mut rc = MatFile::new();
    rc.insert(
        "ph_rc".to_owned(),
        complex32_array(vec![n_ps, n_ifg], corrected.phase_corrected.values),
    );
    if let Some(rereferenced) = corrected.phase_rereferenced {
        rc.insert(
            "ph_reref".to_owned(),
            complex32_array(vec![n_ps, n_ifg], rereferenced.values),
        );
    }
    let mut bp = MatFile::new();
    bp.insert(
        "bperp_mat".to_owned(),
        f32_array(
            vec![n_ps, n_patch_ifg],
            baseline.values.iter().map(|&value| value as f32).collect(),
        ),
    );
    write_mat(transaction.path("ps2.mat"), &ps2).map_err(|error| error.to_string())?;
    write_mat(transaction.path("pm2.mat"), &pm2).map_err(|error| error.to_string())?;
    write_mat(transaction.path("rc2.mat"), &rc).map_err(|error| error.to_string())?;
    write_mat(transaction.path("bp2.mat"), &bp).map_err(|error| error.to_string())?;
    let mut files = vec!["ps2.mat", "pm2.mat", "rc2.mat", "bp2.mat", "psver.mat"];
    write_optional_patch(&transaction, rows, &mut files)?;
    let mut version = MatFile::new();
    version.insert("psver".to_owned(), scalar(2.0));
    write_mat(transaction.path("psver.mat"), &version).map_err(|error| error.to_string())?;
    write_mat(transaction.path("ph2.mat"), &ph2).map_err(|error| error.to_string())?;
    files.push("ph2.mat");
    let mut removals = vec!["inc2.mat"];
    if !rows.iter().all(|row| row.height.is_some()) {
        removals.push("hgt2.mat");
    }
    if !rows.iter().all(|row| row.look_angle.is_some()) {
        removals.push("la2.mat");
    }
    if !rows.iter().all(|row| row.amplitude_dispersion.is_some()) {
        removals.push("da2.mat");
    }
    transaction
        .commit_with_removals(&files, "ph2.mat", &removals)
        .map_err(|error| error.to_string())
}

pub(super) fn base_ps(source: &MatFile, n_ps: usize, n_ifg: usize) -> Result<MatFile, String> {
    let mut output = MatFile::new();
    output.insert(
        "bperp".to_owned(),
        f32_array(vec![n_ifg, 1], numeric_f32(source, "bperp")?),
    );
    output.insert(
        "day".to_owned(),
        f64_array(vec![n_ifg, 1], numeric_f64(source, "day")?),
    );
    output.insert(
        "ll0".to_owned(),
        f64_array(vec![1, 2], numeric_f64(source, "ll0")?),
    );
    for key in ["master_day", "master_ix", "n_ifg", "n_image"] {
        output.insert(key.to_owned(), scalar(numeric_f64(source, key)?[0]));
    }
    output.insert("n_ps".to_owned(), scalar(n_ps as f64));
    for key in ["mean_incidence", "mean_range"] {
        if source.contains_key(key) {
            output.insert(key.to_owned(), scalar(numeric_f64(source, key)?[0]));
        }
    }
    Ok(output)
}

pub(super) fn baseline_matrix(
    rows: &[rustamps_core::stages::stage5::Stage5Row],
    ps: &MatFile,
    n_ifg: usize,
    master_ix: usize,
) -> Result<Matrix<f64>, String> {
    let nominal = numeric_f64(ps, "bperp")?;
    if nominal.len() != n_ifg {
        return Err("ps1.bperp does not match n_ifg".to_owned());
    }
    let nominal = nominal
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| (index + 1 != master_ix).then_some(value))
        .collect::<Vec<_>>();
    let mut values = Vec::with_capacity(rows.len() * (n_ifg - 1));
    for row in rows {
        if let Some(baseline) = &row.bperp {
            if baseline.len() != n_ifg - 1 {
                return Err("bp1 row does not match non-master interferograms".to_owned());
            }
            values.extend(baseline.iter().map(|&value| f64::from(value)));
        } else {
            values.extend_from_slice(&nominal);
        }
    }
    Matrix::new(rows.len(), n_ifg - 1, values).map_err(|error| error.to_string())
}

pub(super) fn write_optional_patch(
    transaction: &StageTransaction,
    rows: &[rustamps_core::stages::stage5::Stage5Row],
    files: &mut Vec<&'static str>,
) -> Result<(), String> {
    let n = rows.len();
    macro_rules! optional {
        ($field:ident, $name:literal, $key:literal, $writer:ident, $cast:expr) => {{
            let values = rows
                .iter()
                .map(|row| row.$field.map($cast))
                .collect::<Option<Vec<_>>>();
            if let Some(values) = values {
                let mut payload = MatFile::new();
                payload.insert($key.to_owned(), $writer(vec![n, 1], values));
                write_mat(transaction.path($name), &payload).map_err(|error| error.to_string())?;
                files.push($name);
            }
        }};
    }
    optional!(height, "hgt2.mat", "hgt", f32_array, |value| value);
    optional!(look_angle, "la2.mat", "la", f64_array, |value| value);
    optional!(amplitude_dispersion, "da2.mat", "D_A", f64_array, |value| {
        value
    });
    Ok(())
}

pub fn merged(
    root: &Path,
    base: &MatFile,
    merged: Stage5Merged,
    master_ix: usize,
    n_ifg: usize,
) -> Result<(), String> {
    super::merged_write::write(root, base, merged, master_ix, n_ifg)
}
