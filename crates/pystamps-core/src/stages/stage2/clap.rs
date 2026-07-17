use super::{ComplexGrid, Stage2Error};
use crate::stages::stage1::Matrix;
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::FftPlanner;

fn fft2(values: &mut [Complex64], rows: usize, cols: usize, inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let row_fft = if inverse {
        planner.plan_fft_inverse(cols)
    } else {
        planner.plan_fft_forward(cols)
    };
    for row in 0..rows {
        row_fft.process(&mut values[row * cols..(row + 1) * cols]);
    }
    let col_fft = if inverse {
        planner.plan_fft_inverse(rows)
    } else {
        planner.plan_fft_forward(rows)
    };
    let mut column = vec![Complex64::new(0.0, 0.0); rows];
    for col in 0..cols {
        for row in 0..rows {
            column[row] = values[row * cols + col];
        }
        col_fft.process(&mut column);
        for row in 0..rows {
            values[row * cols + col] = column[row];
        }
    }
    if inverse {
        let scale = (rows * cols) as f64;
        values.iter_mut().for_each(|value| *value /= scale);
    }
}

fn roll(values: &[f64], rows: usize, cols: usize, row_shift: isize, col_shift: isize) -> Vec<f64> {
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            let source_row = (row as isize - row_shift).rem_euclid(rows as isize) as usize;
            let source_col = (col as isize - col_shift).rem_euclid(cols as isize) as usize;
            output[row * cols + col] = values[source_row * cols + source_col];
        }
    }
    output
}

pub fn clap_gaussian_kernel() -> [f64; 49] {
    let standard_deviation = 6.0 / 5.0;
    let mut gaussian = [0.0; 7];
    for (index, value) in gaussian.iter_mut().enumerate() {
        let x = (index as f64 - 3.0) / standard_deviation;
        *value = (-0.5 * x * x).exp();
    }
    let mut kernel = [0.0; 49];
    for row in 0..7 {
        for col in 0..7 {
            kernel[row * 7 + col] = gaussian[row] * gaussian[col];
        }
    }
    kernel
}

fn convolve(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let kernel = clap_gaussian_kernel();
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            for kernel_row in 0..7 {
                let source_row = row as isize + kernel_row as isize - 3;
                if !(0..rows as isize).contains(&source_row) {
                    continue;
                }
                for kernel_col in 0..7 {
                    let source_col = col as isize + kernel_col as isize - 3;
                    if (0..cols as isize).contains(&source_col) {
                        sum += values[source_row as usize * cols + source_col as usize]
                            * kernel[kernel_row * 7 + kernel_col];
                    }
                }
            }
            output[row * cols + col] = sum;
        }
    }
    output
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

pub fn clap_filter_patch(
    phase: &[Complex64],
    rows: usize,
    cols: usize,
    alpha: f64,
    beta: f64,
    low_pass: &[f64],
) -> Result<Vec<Complex64>, Stage2Error> {
    if rows == 0 || cols == 0 || phase.len() != rows * cols || low_pass.len() != phase.len() {
        return Err(Stage2Error::InvalidInput("CLAP patch shapes do not match"));
    }
    let mut spectrum = phase
        .iter()
        .map(|value| {
            if value.re.is_nan() || value.im.is_nan() {
                Complex64::new(0.0, 0.0)
            } else {
                *value
            }
        })
        .collect::<Vec<_>>();
    fft2(&mut spectrum, rows, cols, false);
    let magnitude = spectrum
        .iter()
        .map(|value| value.norm())
        .collect::<Vec<_>>();
    let shifted = roll(
        &magnitude,
        rows,
        cols,
        (rows / 2) as isize,
        (cols / 2) as isize,
    );
    let smoothed = convolve(&shifted, rows, cols);
    let mut adaptive = roll(
        &smoothed,
        rows,
        cols,
        -((rows / 2) as isize),
        -((cols / 2) as isize),
    );
    let center = median(adaptive.clone());
    if center != 0.0 {
        adaptive.iter_mut().for_each(|value| *value /= center);
    }
    adaptive
        .iter_mut()
        .for_each(|value| *value = (value.powf(alpha) - 1.0).max(0.0));
    for index in 0..spectrum.len() {
        spectrum[index] *= adaptive[index] * beta + low_pass[index];
    }
    fft2(&mut spectrum, rows, cols, true);
    Ok(spectrum)
}

fn window_weight(window: usize, row: usize, col: usize, row_shift: usize, col_shift: usize) -> f64 {
    if row < row_shift || col < col_shift {
        return 0.0;
    }
    let row = row - row_shift;
    let col = col - col_shift;
    if row >= window || col >= window {
        return 0.0;
    }
    row.min(window - 1 - row) as f64 + col.min(window - 1 - col) as f64 + 1e-6
}

fn filter_plane(
    phase: &[Complex64],
    rows: usize,
    cols: usize,
    alpha: f64,
    beta: f64,
    window: usize,
    padding: usize,
    low_pass: &[f64],
) -> Result<Vec<Complex64>, Stage2Error> {
    let increment = (window / 4).max(1);
    let windows_row = rows.div_ceil(increment) as isize - 3;
    let windows_col = cols.div_ceil(increment) as isize - 3;
    let mut output = vec![Complex64::new(0.0, 0.0); rows * cols];
    if windows_row <= 0 || windows_col <= 0 {
        return Ok(output);
    }
    let extended = window + padding;
    let mut patch = vec![Complex64::new(0.0, 0.0); extended * extended];
    for row_window in 0..windows_row as usize {
        let mut row_start = row_window * increment;
        let mut row_end = row_start + window;
        let mut row_shift = 0;
        if row_end > rows {
            row_shift = row_end - rows;
            row_end = rows;
            row_start = rows - window;
        }
        for col_window in 0..windows_col as usize {
            let mut col_start = col_window * increment;
            let mut col_end = col_start + window;
            let mut col_shift = 0;
            if col_end > cols {
                col_shift = col_end - cols;
                col_end = cols;
                col_start = cols - window;
            }
            patch.fill(Complex64::new(0.0, 0.0));
            for row in 0..window {
                for col in 0..window {
                    patch[row * extended + col] = phase[(row_start + row) * cols + col_start + col];
                }
            }
            let filtered = clap_filter_patch(&patch, extended, extended, alpha, beta, low_pass)?;
            for row in 0..row_end - row_start {
                for col in 0..col_end - col_start {
                    output[(row_start + row) * cols + col_start + col] += filtered
                        [row * extended + col]
                        * window_weight(window, row, col, row_shift, col_shift);
                }
            }
        }
    }
    Ok(output)
}

pub fn clap_filter_grid_stack(
    grid: &ComplexGrid,
    alpha: f64,
    beta: f64,
    window: usize,
    padding: usize,
    low_pass: &Matrix<f64>,
) -> Result<ComplexGrid, Stage2Error> {
    let extended = window + padding;
    if window == 0 || window % 2 != 0 || low_pass.rows != extended || low_pass.cols != extended {
        return Err(Stage2Error::InvalidInput(
            "invalid CLAP window or low-pass shape",
        ));
    }
    let planes = (0..grid.planes)
        .into_par_iter()
        .map(|plane| {
            let values = (0..grid.rows * grid.cols)
                .map(|cell| {
                    let value = grid.values[cell * grid.planes + plane];
                    Complex64::new(f64::from(value.re), f64::from(value.im))
                })
                .collect::<Vec<_>>();
            filter_plane(
                &values,
                grid.rows,
                grid.cols,
                alpha,
                beta,
                window,
                padding,
                &low_pass.values,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut values = vec![crate::stages::stage1::Complex32::new(0.0, 0.0); grid.values.len()];
    for plane in 0..grid.planes {
        for cell in 0..grid.rows * grid.cols {
            let value = planes[plane][cell];
            values[cell * grid.planes + plane] =
                crate::stages::stage1::Complex32::new(value.re as f32, value.im as f32);
        }
    }
    Ok(ComplexGrid {
        rows: grid.rows,
        cols: grid.cols,
        planes: grid.planes,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_pass_only_patch_is_an_fft_round_trip() {
        let phase = (0..16)
            .map(|index| Complex64::new(index as f64, -(index as f64)))
            .collect::<Vec<_>>();
        let output = clap_filter_patch(&phase, 4, 4, 2.5, 0.0, &[1.0; 16]).unwrap();
        for (actual, expected) in output.iter().zip(phase) {
            assert!((*actual - expected).norm() < 1e-10);
        }
    }
}
