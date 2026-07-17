mod legacy_guard;
mod metadata;
mod output;
mod raw;

use std::path::Path;

use rustamps_core::stages::stage1::{run_stage1, Matrix, Stage1Input};

use crate::{PipelineError, RunConfig};

use super::{failure, params::Params};

pub fn run(patch: &Path, _config: &RunConfig) -> Result<String, PipelineError> {
    let ij = raw::text_matrix(&patch.join("pscands.1.ij"), 3).map_err(|e| failure(1, e))?;
    let n_ps = ij.len() / 3;
    if n_ps == 0 {
        return Err(failure(1, "pscands.1.ij contains no candidates"));
    }
    let metadata = metadata::resolve(patch, &ij, n_ps).map_err(|e| failure(1, e))?;
    validate_sensor_params(patch, &metadata).map_err(|e| failure(1, e))?;
    let phase = raw::phase_matrix(&patch.join("pscands.1.ph"), n_ps, metadata.days.len())
        .map_err(|e| failure(1, e))?;
    let lonlat = raw::be_f32_matrix(&patch.join("pscands.1.ll"), n_ps, 2)
        .map_err(|e| failure(1, e))?
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    validate_lonlat(&lonlat).map_err(|e| failure(1, e))?;
    let amplitude_dispersion =
        raw::optional_text_vector(&patch.join("pscands.1.da"), n_ps).map_err(|e| failure(1, e))?;
    let height = raw::optional_be_f32_vector(&patch.join("pscands.1.hgt"), n_ps)
        .map_err(|e| failure(1, e))?;

    let result = run_stage1(Stage1Input {
        ij: Matrix::new(n_ps, 3, ij).map_err(|e| failure(1, e))?,
        phase: Matrix::new(n_ps, metadata.days.len(), phase).map_err(|e| failure(1, e))?,
        lonlat: Matrix::new(n_ps, 2, lonlat).map_err(|e| failure(1, e))?,
        day_yyyymmdd: metadata.days,
        master_day_yyyymmdd: metadata.master,
        bperp: metadata.bperp,
        per_pixel_bperp: metadata.bperp_mat,
        amplitude_dispersion,
        height,
        heading_deg: metadata.heading_deg,
    })
    .map_err(|e| failure(1, e))?;
    let count = result.ij.rows;
    output::write(
        patch,
        result,
        metadata.mean_range,
        metadata.mean_incidence,
        metadata.look_angle,
    )
    .map_err(|e| failure(1, e))?;
    Ok(format!("Stage 1 created ps1/ph1 for {count} candidates"))
}

fn validate_lonlat(values: &[f64]) -> Result<(), String> {
    if values.chunks_exact(2).all(|point| {
        point[0].is_finite()
            && (-180.0..=180.0).contains(&point[0])
            && point[1].is_finite()
            && (-90.0..=90.0).contains(&point[1])
    }) {
        Ok(())
    } else {
        Err("pscands.1.ll contains invalid longitude/latitude values; expected big-endian f32 degrees".to_owned())
    }
}

fn validate_sensor_params(patch: &Path, metadata: &metadata::Metadata) -> Result<(), String> {
    if metadata.wavelength.is_none() {
        return Ok(());
    }
    let params = Params::load(patch)?;
    if !params.contains("heading") || !params.contains("lambda") {
        return Err(
            "SNAP Stage 1 requires heading and lambda in parms.mat; rerun `rustamps prep snap`"
                .to_owned(),
        );
    }
    let heading = params.scalar("heading", f64::NAN)?;
    let wavelength = params.scalar("lambda", f64::NAN)?;
    if !heading.is_finite() || !wavelength.is_finite() || wavelength <= 0.0 {
        return Err("parms.mat heading/lambda must be finite and lambda positive".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_lonlat;

    #[test]
    fn longitude_latitude_validation_rejects_non_finite_or_out_of_range_values() {
        assert!(validate_lonlat(&[12.0, 45.0, -180.0, 90.0]).is_ok());
        assert!(validate_lonlat(&[f64::INFINITY, 45.0]).is_err());
        assert!(validate_lonlat(&[12.0, 91.0]).is_err());
    }
}
