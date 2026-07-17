use num_complex::{Complex32, Complex64};
use pystamps_core::stages::stage2::clap_filter_patch;

pub fn goldstein_global(
    input: &[Complex32],
    rows: usize,
    cols: usize,
    window: usize,
    alpha: f64,
) -> Result<Vec<Complex32>, String> {
    if input.len() != rows * cols || window == 0 || window % 2 != 0 {
        return Err("invalid Goldstein filter shape or window".to_owned());
    }
    if rows < window || cols < window {
        return Err(format!(
            "minimum resampled grid dimension ({}) is smaller than prefilter window ({window})",
            rows.min(cols)
        ));
    }
    let padding = window / 4;
    let extended = window + padding;
    let increment = (window / 2).max(1);
    let row_windows = rows.div_ceil(increment).saturating_sub(1).max(1);
    let col_windows = cols.div_ceil(increment).saturating_sub(1).max(1);
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
    let mut patch = vec![Complex64::new(0.0, 0.0); extended * extended];
    let pass = vec![1.0_f64; patch.len()];
    for row_window in 0..row_windows {
        let (row_start, row_end, row_shift) = bounds(row_window, increment, window, rows);
        for col_window in 0..col_windows {
            let (col_start, col_end, col_shift) = bounds(col_window, increment, window, cols);
            patch.fill(Complex64::new(0.0, 0.0));
            for row in 0..window {
                for col in 0..window {
                    let value = input[(row_start + row) * cols + col_start + col];
                    patch[row * extended + col] = Complex64::new(value.re.into(), value.im.into());
                }
            }
            // The core CLAP spectrum is a conservative Goldstein equivalent:
            // it preserves weak bins instead of attenuating them below unity.
            let filtered = clap_filter_patch(&patch, extended, extended, alpha, 1.0, &pass)
                .map_err(|error| error.to_string())?;
            for row in 0..row_end - row_start {
                for col in 0..col_end - col_start {
                    let weight = weight(window, row, col, row_shift, col_shift);
                    output[(row_start + row) * cols + col_start + col] +=
                        filtered[row * extended + col] * weight;
                }
            }
        }
    }
    Ok(output
        .into_iter()
        .zip(input)
        .map(|(filtered, source)| {
            let magnitude = source.norm();
            let angle = filtered.im.atan2(filtered.re) as f32;
            Complex32::new(magnitude * angle.cos(), magnitude * angle.sin())
        })
        .collect())
}

fn bounds(block: usize, increment: usize, window: usize, extent: usize) -> (usize, usize, usize) {
    let mut start = block * increment;
    let mut end = start + window;
    let mut shift = 0;
    if end > extent {
        shift = end - extent;
        end = extent;
        start = extent - window;
    }
    (start, end, shift)
}

fn weight(window: usize, row: usize, col: usize, row_shift: usize, col_shift: usize) -> f64 {
    if row < row_shift || col < col_shift {
        return 0.0;
    }
    let row = row - row_shift;
    let col = col - col_shift;
    if row >= window || col >= window {
        return 0.0;
    }
    row.min(window - 1 - row) as f64 + col.min(window - 1 - col) as f64 + 2.0
}
