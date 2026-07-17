use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use super::SnapPrepError;

pub struct CandidateData {
    pub selected: Vec<(usize, usize)>,
    pub lon: Vec<f32>,
    pub lat: Vec<f32>,
    pub height: Vec<f32>,
    pub dispersion: Vec<f32>,
    pub mean_amplitude: Vec<f32>,
}

pub fn candidate_statistics(
    rslc: &[PathBuf],
    lon_path: &Path,
    lat_path: &Path,
    height_path: &Path,
    cells: usize,
    width: usize,
    threshold: f64,
) -> Result<CandidateData, SnapPrepError> {
    let mut sum_amplitude = vec![0.0_f64; cells];
    let mut sum_squares = vec![0.0_f64; cells];
    let mut low_amplitude = vec![false; cells];
    for path in rslc {
        let amplitudes = read_complex_amplitude(path, cells)?;
        let (calibration_sum, calibration_count) = amplitudes
            .par_iter()
            .filter(|value| **value > 0.001)
            .fold(
                || (0.0_f64, 0_usize),
                |(sum, count), value| (sum + f64::from(*value), count + 1),
            )
            .reduce(
                || (0.0, 0),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        let calibration = if calibration_count == 0 {
            0.0
        } else {
            calibration_sum / calibration_count as f64
        };
        sum_amplitude
            .par_iter_mut()
            .zip(sum_squares.par_iter_mut())
            .zip(low_amplitude.par_iter_mut())
            .zip(amplitudes.par_iter())
            .for_each(|(((sum, squares), has_low), raw)| {
                let normalized = f64::from(*raw) / calibration;
                if normalized <= 0.00005 {
                    *has_low = true;
                    *sum = 0.0;
                } else {
                    *sum += normalized;
                    *squares += normalized * normalized;
                }
            });
    }
    let lon = read_be_f32(lon_path, cells)?;
    let lat = read_be_f32(lat_path, cells)?;
    let height = read_be_f32(height_path, cells)?;
    let acquisition_count = rslc.len() as f64;
    let mut dispersion = vec![0.0_f32; cells];
    let mut mean_amplitude = vec![0.0_f32; cells];
    dispersion
        .par_iter_mut()
        .zip(mean_amplitude.par_iter_mut())
        .enumerate()
        .for_each(|(index, (da, mean))| {
            *da = (acquisition_count * sum_squares[index]
                / (sum_amplitude[index] * sum_amplitude[index])
                - 1.0)
                .max(0.0)
                .sqrt() as f32;
            *mean = sum_amplitude[index] as f32;
        });
    let selected = (0..cells)
        .filter(|&index| {
            let exact_dispersion = (acquisition_count * sum_squares[index]
                / (sum_amplitude[index] * sum_amplitude[index])
                - 1.0)
                .max(0.0)
                .sqrt();
            lon[index].is_finite()
                && lat[index].is_finite()
                && height[index].is_finite()
                && !low_amplitude[index]
                && exact_dispersion.is_finite()
                && mean_amplitude[index] > 0.0
                && exact_dispersion < threshold
        })
        .collect::<Vec<_>>();
    Ok(CandidateData {
        selected: selected
            .into_iter()
            .map(|index| (index / width, index % width))
            .collect(),
        lon,
        lat,
        height,
        dispersion,
        mean_amplitude,
    })
}

fn read_complex_amplitude(path: &Path, cells: usize) -> Result<Vec<f32>, SnapPrepError> {
    require_size(path, cells * 8)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut bytes = [0_u8; 8];
    let mut output = Vec::with_capacity(cells);
    for _ in 0..cells {
        reader.read_exact(&mut bytes)?;
        let real = f32::from_be_bytes(bytes[..4].try_into().unwrap());
        let imag = f32::from_be_bytes(bytes[4..].try_into().unwrap());
        output.push(real.hypot(imag));
    }
    Ok(output)
}

fn read_be_f32(path: &Path, cells: usize) -> Result<Vec<f32>, SnapPrepError> {
    require_size(path, cells * 4)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut bytes = [0_u8; 4];
    let mut output = Vec::with_capacity(cells);
    for _ in 0..cells {
        reader.read_exact(&mut bytes)?;
        output.push(f32::from_be_bytes(bytes));
    }
    Ok(output)
}

fn require_size(path: &Path, expected: usize) -> Result<(), SnapPrepError> {
    let actual = fs::metadata(path)?.len() as usize;
    if actual != expected {
        Err(SnapPrepError::Invalid(format!(
            "unexpected raster size for {}: expected {expected}, found {actual}",
            path.display()
        )))
    } else {
        Ok(())
    }
}
