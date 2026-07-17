use super::{Stage2Error, Stage2Input};
use crate::stages::stage1::{Complex32, Matrix};

#[derive(Clone, Debug, PartialEq)]
pub struct ComplexGrid {
    pub rows: usize,
    pub cols: usize,
    pub planes: usize,
    pub values: Vec<Complex32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridLayout {
    pub indices: Vec<[usize; 2]>,
    pub rows: usize,
    pub cols: usize,
}

impl GridLayout {
    pub fn linear_indices(&self) -> Vec<usize> {
        self.indices
            .iter()
            .map(|[row, col]| row * self.cols + col)
            .collect()
    }
}

pub fn non_master_phase(input: &Stage2Input) -> Matrix<Complex32> {
    if input.small_baseline {
        return input.phase.clone();
    }
    let master = input.master_ix - 1;
    let mut values = Vec::with_capacity(input.phase.rows * (input.phase.cols - 1));
    for row in 0..input.phase.rows {
        for col in 0..input.phase.cols {
            if col != master {
                values.push(input.phase.row(row)[col]);
            }
        }
    }
    Matrix {
        rows: input.phase.rows,
        cols: input.phase.cols - 1,
        values,
    }
}

pub fn normalize_phase_matrix(phase: &Matrix<Complex32>) -> (Matrix<Complex32>, Matrix<f32>) {
    let mut normalized = Vec::with_capacity(phase.values.len());
    let mut amplitude = Vec::with_capacity(phase.values.len());
    for &value in &phase.values {
        let mut magnitude = value.norm();
        if magnitude == 0.0 {
            magnitude = 1.0;
        }
        normalized.push(value / magnitude);
        amplitude.push(magnitude);
    }
    (
        Matrix {
            rows: phase.rows,
            cols: phase.cols,
            values: normalized,
        },
        Matrix {
            rows: phase.rows,
            cols: phase.cols,
            values: amplitude,
        },
    )
}

pub fn phase_weight_block(
    phase: &Matrix<Complex32>,
    bperp: &Matrix<f64>,
    k_ps: &[f64],
    weighting: &[f64],
) -> Result<Matrix<Complex32>, Stage2Error> {
    if bperp.rows != phase.rows
        || bperp.cols != phase.cols
        || k_ps.len() != phase.rows
        || weighting.len() != phase.rows
    {
        return Err(Stage2Error::InvalidInput(
            "phase-weight shapes do not match",
        ));
    }
    let mut values = Vec::with_capacity(phase.values.len());
    for row in 0..phase.rows {
        for col in 0..phase.cols {
            let index = row * phase.cols + col;
            let (sin, cos) = (bperp.values[index] * k_ps[row]).sin_cos();
            let value = phase.values[index];
            values.push(Complex32::new(
                ((f64::from(value.re) * cos + f64::from(value.im) * sin) * weighting[row]) as f32,
                ((f64::from(value.im) * cos - f64::from(value.re) * sin) * weighting[row]) as f32,
            ));
        }
    }
    Ok(Matrix {
        rows: phase.rows,
        cols: phase.cols,
        values,
    })
}

pub fn grid_indices(xy: &Matrix<f32>, grid_size: f32) -> Result<GridLayout, Stage2Error> {
    if xy.rows == 0 || xy.cols != 3 || !grid_size.is_finite() || grid_size == 0.0 {
        return Err(Stage2Error::InvalidInput("invalid grid geometry"));
    }
    let x_min = (0..xy.rows)
        .map(|row| xy.row(row)[1])
        .fold(f32::INFINITY, f32::min);
    let y_min = (0..xy.rows)
        .map(|row| xy.row(row)[2])
        .fold(f32::INFINITY, f32::min);
    let mut one_based = (0..xy.rows)
        .map(|row| {
            [
                ((xy.row(row)[2] - y_min + 1e-6) / grid_size).ceil() as usize,
                ((xy.row(row)[1] - x_min + 1e-6) / grid_size).ceil() as usize,
            ]
        })
        .collect::<Vec<_>>();
    let max_row = one_based.iter().map(|value| value[0]).max().unwrap();
    let max_col = one_based.iter().map(|value| value[1]).max().unwrap();
    for value in &mut one_based {
        if max_row > 1 && value[0] == max_row {
            value[0] -= 1;
        }
        if max_col > 1 && value[1] == max_col {
            value[1] -= 1;
        }
        value[0] = value[0].max(1);
        value[1] = value[1].max(1);
    }
    let rows = one_based.iter().map(|value| value[0]).max().unwrap();
    let cols = one_based.iter().map(|value| value[1]).max().unwrap();
    Ok(GridLayout {
        indices: one_based
            .into_iter()
            .map(|[row, col]| [row - 1, col - 1])
            .collect(),
        rows,
        cols,
    })
}

pub fn accumulate_weighted_grid(
    phase: &Matrix<Complex32>,
    layout: &GridLayout,
) -> Result<ComplexGrid, Stage2Error> {
    if phase.rows != layout.indices.len() {
        return Err(Stage2Error::InvalidInput(
            "grid index count must match phase rows",
        ));
    }
    let mut values = vec![Complex32::new(0.0, 0.0); layout.rows * layout.cols * phase.cols];
    for row in 0..phase.rows {
        let [grid_row, grid_col] = layout.indices[row];
        for col in 0..phase.cols {
            values[(grid_row * layout.cols + grid_col) * phase.cols + col] += phase.row(row)[col];
        }
    }
    Ok(ComplexGrid {
        rows: layout.rows,
        cols: layout.cols,
        planes: phase.cols,
        values,
    })
}

pub fn histogram_with_centers(values: &[f64], centers: &[f64]) -> Vec<f64> {
    if centers.is_empty() {
        return Vec::new();
    }
    if centers.len() == 1 {
        return vec![values.iter().filter(|v| v.is_finite()).count() as f64];
    }
    let mids = centers
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect::<Vec<_>>();
    let mut counts = vec![0.0; centers.len()];
    for &value in values.iter().filter(|value| value.is_finite()) {
        counts[mids.partition_point(|mid| *mid < value)] += 1.0;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_matches_matlab_one_based_edge_collapse() {
        let xy = Matrix::new(3, 3, vec![1.0, 0.0, 0.0, 2.0, 10.0, 10.0, 3.0, 20.0, 20.0]).unwrap();
        let grid = grid_indices(&xy, 10.0).unwrap();
        assert_eq!(grid.indices, vec![[0, 0], [1, 1], [1, 1]]);
    }

    #[test]
    fn histogram_assigns_midpoint_ties_to_lower_bin() {
        assert_eq!(
            histogram_with_centers(&[0.5, 1.0, 1.5], &[0.0, 1.0, 2.0]),
            vec![1.0, 2.0, 0.0]
        );
    }
}
