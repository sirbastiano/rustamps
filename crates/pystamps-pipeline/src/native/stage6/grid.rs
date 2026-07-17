use std::path::Path;

use super::filter::goldstein_global;
use super::grid_checkpoint;
use super::input::Input;
use num_complex::Complex32;
use pystamps_core::stage6::grid_accumulate;

pub struct Grid {
    pub fingerprint: u64,
    pub rows: usize,
    pub cols: usize,
    pub mask: Vec<bool>,
    pub coordinates: Vec<[usize; 2]>,
    pub phase: Vec<Complex32>,
    pub phase_in: Vec<Complex32>,
    pub n_points: usize,
    pub min_x: f32,
    pub min_y: f32,
}

pub fn load_or_build(root: &Path, input: &Input) -> Result<Grid, String> {
    let path = root.join("uw_grid.mat");
    if path.is_file() {
        if let Some(grid) = grid_checkpoint::load(&path, input)? {
            return Ok(grid);
        }
    }
    let grid = build(input)?;
    grid_checkpoint::write(&path, input, &grid)?;
    Ok(grid)
}

fn build(input: &Input) -> Result<Grid, String> {
    let grid_size = input.options.grid_size as f32;
    let min_x = input
        .xy
        .iter()
        .map(|point| point[0] as f32)
        .fold(f32::INFINITY, f32::min);
    let min_y = input
        .xy
        .iter()
        .map(|point| point[1] as f32)
        .fold(f32::INFINITY, f32::min);
    let mut coordinates = input
        .xy
        .iter()
        .map(|point| {
            [
                (((point[1] as f32 - min_y + 1.0e-3) / grid_size).ceil() as usize).max(1),
                (((point[0] as f32 - min_x + 1.0e-3) / grid_size).ceil() as usize).max(1),
            ]
        })
        .collect::<Vec<_>>();
    collapse_maximum(&mut coordinates, 0);
    collapse_maximum(&mut coordinates, 1);
    let rows = coordinates.iter().map(|point| point[0]).max().unwrap_or(1);
    let cols = coordinates.iter().map(|point| point[1]).max().unwrap_or(1);
    for point in &mut coordinates {
        point[0] -= 1;
        point[1] -= 1;
    }
    let phase_in = select_phase(input);
    let cells = coordinates
        .iter()
        .map(|point| point[1] * rows + point[0])
        .collect::<Vec<_>>();
    let accumulated = grid_accumulate(
        &phase_in,
        input.n_ps,
        input.unwrap.len(),
        &cells,
        rows * cols,
    )
    .map_err(|error| error.to_string())?;
    let mut mask = vec![false; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            mask[row * cols + col] =
                accumulated[(col * rows + row) * input.unwrap.len()] != Complex32::new(0.0, 0.0);
        }
    }
    let n_points = mask.iter().filter(|&&value| value).count();
    if n_points == 0 {
        return Err("uw_grid has no non-zero points in its first interferogram".to_owned());
    }
    let mut phase = vec![Complex32::new(0.0, 0.0); n_points * input.unwrap.len()];
    for ifg in 0..input.unwrap.len() {
        let mut plane = vec![Complex32::new(0.0, 0.0); rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                plane[row * cols + col] =
                    accumulated[(col * rows + row) * input.unwrap.len() + ifg];
            }
        }
        if input.options.prefilter {
            plane = goldstein_global(
                &plane,
                rows,
                cols,
                input.options.filter_window,
                input.options.filter_alpha,
            )?;
        }
        let mut output_row = 0;
        for col in 0..cols {
            for row in 0..rows {
                if mask[row * cols + col] {
                    phase[output_row * input.unwrap.len() + ifg] = plane[row * cols + col];
                    output_row += 1;
                }
            }
        }
    }
    Ok(Grid {
        fingerprint: input.fingerprint,
        rows,
        cols,
        mask,
        coordinates,
        phase,
        phase_in,
        n_points,
        min_x,
        min_y,
    })
}

pub(super) fn select_phase(input: &Input) -> Vec<Complex32> {
    (0..input.n_ps)
        .flat_map(|row| {
            input
                .unwrap
                .iter()
                .map(move |&col| input.phase[row * input.n_ifg + col])
        })
        .collect()
}

fn collapse_maximum(points: &mut [[usize; 2]], axis: usize) {
    let maximum = points.iter().map(|point| point[axis]).max().unwrap_or(1);
    if maximum > 1 {
        for point in points {
            if point[axis] == maximum {
                point[axis] -= 1;
            }
        }
    }
}
