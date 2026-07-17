use std::collections::BTreeMap;
use std::path::Path;

use num_complex::{Complex32, Complex64};
use rustamps_core::stage6::estimate_la_error_single_master;
use rustamps_io::{read_mat, write_mat};

use super::super::mat::{f32_array, f64_array, numeric_f32, shape};
use super::cache_meta::{self, Checksum};
use super::grid::Grid;
use super::input::Input;
use super::interp::Interpolation;

pub struct SpaceTime {
    pub noise: Vec<f32>,
    pub unwrapped: Vec<f32>,
}

pub fn load_or_build(
    root: &Path,
    input: &Input,
    grid: &Grid,
    interp: &Interpolation,
) -> Result<SpaceTime, String> {
    let path = root.join("uw_space_time.mat");
    if path.is_file() {
        if let Some(result) = load(&path, input, interp.edges.len())? {
            return Ok(result);
        }
    }
    let result = build(input, grid, interp)?;
    write(&path, input, &result, interp.edges.len())?;
    Ok(result)
}

fn load(path: &Path, input: &Input, edges: usize) -> Result<Option<SpaceTime>, String> {
    let file = read_mat(path).map_err(|error| error.to_string())?;
    if !cache_meta::matches(&file, input.fingerprint)? {
        return Ok(None);
    }
    let expected = [edges, input.unwrap.len()];
    if shape(&file, "dph_noise")? != expected || shape(&file, "dph_space_uw")? != expected {
        return Err("uw_space_time checkpoint does not match uw_grid/uw_interp".to_owned());
    }
    let result = SpaceTime {
        noise: numeric_f32(&file, "dph_noise")?,
        unwrapped: numeric_f32(&file, "dph_space_uw")?,
    };
    cache_meta::validate(
        &file,
        payload_checksum(input.fingerprint, edges, &result),
        "uw_space_time",
    )?;
    Ok(Some(result))
}

fn build(input: &Input, grid: &Grid, interp: &Interpolation) -> Result<SpaceTime, String> {
    let n_ifg = input.unwrap.len();
    let mut spatial = Vec::with_capacity(interp.edges.len() * n_ifg);
    for edge in &interp.edges {
        for ifg in 0..n_ifg {
            let mut value =
                grid.phase[edge[1] * n_ifg + ifg] * grid.phase[edge[0] * n_ifg + ifg].conj();
            let magnitude = value.norm();
            if magnitude != 0.0 {
                value /= magnitude;
            }
            spatial.push(value);
        }
    }
    let day = input
        .unwrap
        .iter()
        .map(|&index| input.day[index] - input.day[input.master])
        .collect::<Vec<_>>();
    let bperp = input
        .unwrap
        .iter()
        .map(|&index| input.bperp[index])
        .collect::<Vec<_>>();
    let look_angle = estimate_la_error_single_master(
        &spatial,
        interp.edges.len(),
        n_ifg,
        &day,
        &bperp,
        input.options.trial_wraps,
    )
    .map_err(|error| error.to_string())?;
    for edge in 0..interp.edges.len() {
        for ifg in 0..n_ifg {
            let angle = -(f64::from(look_angle[edge]) * bperp[ifg]);
            spatial[edge * n_ifg + ifg] *= Complex32::new(angle.cos() as f32, angle.sin() as f32);
        }
    }
    let close = close_master_indices(&day);
    let mut smooth = Vec::with_capacity(spatial.len());
    let mut noise = Vec::with_capacity(spatial.len());
    for row in spatial.chunks_exact(n_ifg) {
        let (smooth_row, noise_row) = smooth_row(row, &day, input.options.time_window, &close);
        smooth.extend(smooth_row);
        noise.extend(noise_row);
    }
    let mut unwrapped = vec![0.0_f32; spatial.len()];
    for edge in 0..interp.edges.len() {
        let deviation = standard_deviation(&noise[edge * n_ifg..(edge + 1) * n_ifg]);
        // uw_unwrap_space_time.m rejects single-master edge noise above 1.3 rad.
        if deviation > 1.3 {
            noise[edge * n_ifg..(edge + 1) * n_ifg].fill(f32::NAN);
        }
        for ifg in 0..n_ifg {
            let index = edge * n_ifg + ifg;
            unwrapped[index] = smooth[index] + noise[index] + look_angle[edge] * bperp[ifg] as f32;
        }
    }
    Ok(SpaceTime { noise, unwrapped })
}

fn smooth_row(
    row: &[Complex32],
    day: &[f64],
    time_window: f64,
    close: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let count = row.len();
    let row64 = row
        .iter()
        .map(|value| Complex64::new(value.re.into(), value.im.into()))
        .collect::<Vec<_>>();
    let angles = row64.iter().map(|value| value.arg()).collect::<Vec<_>>();
    let mut phasors = vec![Complex64::new(0.0, 0.0); count];
    for output in 0..count {
        let difference = day
            .iter()
            .map(|value| day[output] - value)
            .collect::<Vec<_>>();
        let mut weights = difference
            .iter()
            .map(|value| (-(value * value) / (2.0 * time_window.max(1.0e-6).powi(2))).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>().max(1.0e-12);
        weights.iter_mut().for_each(|value| *value /= total);
        let mean = row64
            .iter()
            .zip(&weights)
            .map(|(value, weight)| *value * *weight)
            .sum::<Complex64>();
        let mean_angle = mean.arg();
        let adjusted = angles
            .iter()
            .zip(&difference)
            .map(|(angle, delta)| {
                let mut value = wrap_pi(*angle - mean_angle);
                if (value + std::f64::consts::PI).abs() <= 2.0e-7 && *delta > 0.0 {
                    value = std::f64::consts::PI;
                }
                value
            })
            .collect::<Vec<_>>();
        let intercept = weighted_intercept(&difference, &adjusted, &weights);
        phasors[output] = mean * Complex64::from_polar(1.0, intercept);
    }
    let noise = row64
        .iter()
        .zip(&phasors)
        .map(|(value, smooth)| (*value * smooth.conj()).arg() as f32)
        .collect::<Vec<_>>();
    let phasors32 = phasors
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let mut smooth = vec![0.0_f32; count];
    if count > 0 {
        smooth[0] = phasors32[0].arg();
        for index in 1..count {
            smooth[index] =
                smooth[index - 1] + (phasors32[index] * phasors32[index - 1].conj()).arg();
        }
        if !close.is_empty() {
            let mean = close.iter().map(|&index| smooth[index]).sum::<f32>() / close.len() as f32;
            let adjustment = mean - mean.sin().atan2(mean.cos());
            smooth.iter_mut().for_each(|value| *value -= adjustment);
        }
    }
    (smooth, noise)
}

fn weighted_intercept(x: &[f64], y: &[f64], weights: &[f64]) -> f64 {
    let s0 = weights.iter().sum::<f64>();
    let s1 = weights.iter().zip(x).map(|(w, x)| w * x).sum::<f64>();
    let s2 = weights.iter().zip(x).map(|(w, x)| w * x * x).sum::<f64>();
    let wy0 = weights.iter().zip(y).map(|(w, y)| w * y).sum::<f64>();
    let wy1 = weights
        .iter()
        .zip(x)
        .zip(y)
        .map(|((w, x), y)| w * x * y)
        .sum::<f64>();
    let determinant = s0 * s2 - s1 * s1;
    if determinant == 0.0 {
        if s0 == 0.0 {
            0.0
        } else {
            wy0 / s0
        }
    } else {
        (wy0 * s2 - wy1 * s1) / determinant
    }
}

fn close_master_indices(day: &[f64]) -> Vec<usize> {
    let insertion = day
        .iter()
        .enumerate()
        .filter(|(_, value)| **value > 0.0)
        .min_by(|left, right| left.1.total_cmp(right.1))
        .map_or(day.len().saturating_sub(1), |(index, _)| index);
    if insertion > 0 {
        vec![insertion - 1, insertion]
    } else {
        vec![insertion]
    }
}

fn standard_deviation(values: &[f32]) -> f64 {
    let mean =
        values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len().max(1) as f64;
    let divisor = values
        .len()
        .saturating_sub(usize::from(values.len() > 1))
        .max(1) as f64;
    (values
        .iter()
        .map(|&value| (f64::from(value) - mean).powi(2))
        .sum::<f64>()
        / divisor)
        .sqrt()
}

fn wrap_pi(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

fn write(path: &Path, input: &Input, result: &SpaceTime, edges: usize) -> Result<(), String> {
    let mut file = BTreeMap::new();
    cache_meta::insert(
        &mut file,
        input.fingerprint,
        payload_checksum(input.fingerprint, edges, result),
    );
    let mut design = vec![0.0_f64; input.unwrap.len() * input.n_ifg];
    for (row, &slave) in input.unwrap.iter().enumerate() {
        design[row * input.n_ifg + input.master] = -1.0;
        design[row * input.n_ifg + slave] = 1.0;
    }
    file.insert(
        "G".to_owned(),
        f64_array(vec![input.unwrap.len(), input.n_ifg], design),
    );
    file.insert(
        "dph_noise".to_owned(),
        f32_array(vec![edges, input.unwrap.len()], result.noise.clone()),
    );
    file.insert(
        "dph_space_uw".to_owned(),
        f32_array(vec![edges, input.unwrap.len()], result.unwrapped.clone()),
    );
    write_mat(path, &file).map_err(|error| error.to_string())
}

fn payload_checksum(fingerprint: u64, edges: usize, result: &SpaceTime) -> u64 {
    let mut checksum = Checksum::new(fingerprint);
    checksum.usize(edges);
    checksum.usize(result.noise.len());
    for &value in &result.noise {
        checksum.f32(value);
    }
    checksum.usize(result.unwrapped.len());
    for &value in &result.unwrapped {
        checksum.f32(value);
    }
    checksum.finish()
}
