use super::types::{ReestimatedSelection, Stage3Error, Stage3Input, Stage3Output};
use crate::stages::stage1::Matrix;

fn select_rows<T: Copy>(matrix: &Matrix<T>, rows: &[usize]) -> Matrix<T> {
    let mut values = Vec::with_capacity(rows.len() * matrix.cols);
    for &row in rows {
        values.extend_from_slice(matrix.row(row));
    }
    Matrix {
        rows: rows.len(),
        cols: matrix.cols,
        values,
    }
}

pub fn da_bin_edges(values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if values.len() < 10_000 {
        return (vec![0.0, 1.0], vec![1.0; values.len()]);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let bin_size = if values.len() >= 50_000 {
        10_000
    } else {
        2_000
    };
    let mut edges = vec![0.0];
    let mut one_based = bin_size;
    while one_based <= values.len() - bin_size {
        edges.push(sorted[one_based - 1]);
        one_based += bin_size;
    }
    edges.push(*sorted.last().unwrap());
    (edges, values.to_vec())
}

pub fn initial_selection(input: &Stage3Input) -> Result<Stage3Output, Stage3Error> {
    let n_ps = input.coherence.len();
    if input.k_ps.len() != n_ps
        || input.c_ps.len() != n_ps
        || input.amplitude_dispersion.len() != n_ps
        || input.coherence_threshold.len() != n_ps
        || input.phase_patch.rows != n_ps
        || input.phase_residual.rows != n_ps
    {
        return Err(Stage3Error::InvalidInput(
            "row-aligned arrays must have equal length",
        ));
    }
    let rows = (0..n_ps)
        .filter(|&row| input.coherence[row] > input.coherence_threshold[row])
        .collect::<Vec<_>>();
    Ok(Stage3Output {
        selected_ix: rows.iter().map(|row| row + 1).collect(),
        keep_ix: vec![true; rows.len()],
        coherence: rows.iter().map(|&row| input.coherence[row]).collect(),
        k_ps: rows.iter().map(|&row| input.k_ps[row]).collect(),
        c_ps: rows.iter().map(|&row| input.c_ps[row]).collect(),
        phase_patch: select_rows(&input.phase_patch, &rows),
        phase_residual: select_rows(&input.phase_residual, &rows),
        coherence_threshold: rows
            .iter()
            .map(|&row| input.coherence_threshold[row])
            .collect(),
    })
}

pub fn apply_reestimate(
    original_k: &[f64],
    reestimated: ReestimatedSelection,
) -> Result<Stage3Output, Stage3Error> {
    let n = reestimated.source_rows.len();
    if reestimated.coherence.len() != n
        || reestimated.k_ps.len() != n
        || reestimated.c_ps.len() != n
        || reestimated.coherence_threshold.len() != n
        || reestimated.phase_patch.rows != n
        || reestimated.phase_residual.rows != n
        || reestimated.bperp_range <= 0.0
    {
        return Err(Stage3Error::InvalidInput("invalid re-estimation payload"));
    }
    let tolerance = 2.0 * std::f64::consts::PI / reestimated.bperp_range;
    let keep_ix = reestimated
        .source_rows
        .iter()
        .enumerate()
        .map(|(row, &source)| {
            source < original_k.len()
                && reestimated.coherence[row] > reestimated.coherence_threshold[row].max(0.0)
                && (original_k[source] - reestimated.k_ps[row]).abs() < tolerance
        })
        .collect();
    Ok(Stage3Output {
        selected_ix: reestimated.source_rows.iter().map(|row| row + 1).collect(),
        keep_ix,
        coherence: reestimated.coherence,
        k_ps: reestimated.k_ps,
        c_ps: reestimated.c_ps,
        phase_patch: reestimated.phase_patch,
        phase_residual: reestimated.phase_residual,
        coherence_threshold: reestimated.coherence_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::stage1::{Complex32, Matrix};

    #[test]
    fn matlab_da_edges_use_one_based_bin_boundaries() {
        let values = (1..=50_000).map(|value| value as f64).collect::<Vec<_>>();
        let (edges, _) = da_bin_edges(&values);
        assert_eq!(
            edges,
            vec![0.0, 10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0]
        );
    }

    #[test]
    fn initial_selection_is_strictly_greater_than_threshold() {
        let input = Stage3Input {
            coherence: vec![0.5, 0.6],
            k_ps: vec![0.0; 2],
            c_ps: vec![0.0; 2],
            amplitude_dispersion: vec![1.0; 2],
            phase_patch: Matrix::new(2, 1, vec![Complex32::new(1.0, 0.0); 2]).unwrap(),
            phase_residual: Matrix::new(2, 1, vec![0.0; 2]).unwrap(),
            coherence_threshold: vec![0.5, 0.5],
        };
        assert_eq!(initial_selection(&input).unwrap().selected_ix, vec![2]);
    }
}
