use super::qr::QrSolver;
use super::Stage7Error;

#[derive(Clone, Debug, PartialEq)]
pub struct DerampOutputs {
    pub phase: Vec<f64>,
    pub ramp: Vec<f64>,
}

pub fn center_to_reference(
    phase: &[f64],
    n_ps: usize,
    n_ifg: usize,
    reference_indices: &[usize],
) -> Result<Vec<f64>, Stage7Error> {
    validate_phase_shape(phase, n_ps, n_ifg)?;
    if reference_indices.iter().any(|&index| index >= n_ps) {
        return Err(Stage7Error::new(
            "reference index is outside the phase matrix",
        ));
    }
    if reference_indices.is_empty() {
        return Ok(phase.to_vec());
    }

    let mut means = vec![f64::NAN; n_ifg];
    for column in 0..n_ifg {
        let mut sum = 0.0;
        let mut count = 0;
        for &row in reference_indices {
            let value = phase[row * n_ifg + column];
            if !value.is_nan() {
                sum += value;
                count += 1;
            }
        }
        if count > 0 {
            means[column] = sum / count as f64;
        }
    }
    Ok(phase
        .iter()
        .enumerate()
        .map(|(index, value)| value - means[index % n_ifg])
        .collect())
}

pub fn deramp_phase(
    phase: &[f64],
    xy: &[f64],
    n_ps: usize,
    n_ifg: usize,
) -> Result<DerampOutputs, Stage7Error> {
    validate_phase_shape(phase, n_ps, n_ifg)?;
    if xy.len() != n_ps * 2 || xy.iter().any(|value| !value.is_finite()) {
        return Err(Stage7Error::new(
            "deramp xy must be a finite n_ps by 2 matrix",
        ));
    }
    if phase.iter().any(|value| value.is_infinite()) {
        return Err(Stage7Error::new("deramp phase contains infinity"));
    }

    let full_design = build_design(xy, 0..n_ps);
    let shared_solver = if phase.iter().any(|value| value.is_nan()) {
        None
    } else {
        Some(QrSolver::factor(&full_design, n_ps, 3).map_err(Stage7Error::new)?)
    };
    let mut output = phase.to_vec();
    let mut ramp = vec![f64::NAN; phase.len()];

    for column in 0..n_ifg {
        let valid = if shared_solver.is_some() {
            None
        } else {
            Some(
                (0..n_ps)
                    .filter(|&row| !phase[row * n_ifg + column].is_nan())
                    .collect::<Vec<_>>(),
            )
        };
        if valid.as_ref().is_some_and(|rows| rows.len() <= 5) {
            continue;
        }

        let coefficients = if let Some(solver) = shared_solver.as_ref() {
            solver.solve_with(|row| phase[row * n_ifg + column])
        } else {
            let valid_rows = valid.as_ref().expect("masked fit has valid rows");
            let design = build_design(xy, valid_rows.iter().copied());
            let solver =
                QrSolver::factor(&design, valid_rows.len(), 3).map_err(Stage7Error::new)?;
            solver.solve_with(|local_row| phase[valid_rows[local_row] * n_ifg + column])
        };

        for row in 0..n_ps {
            let fitted = coefficients[0] * (xy[row * 2] / 1000.0)
                + coefficients[1] * (xy[row * 2 + 1] / 1000.0)
                + coefficients[2];
            let index = row * n_ifg + column;
            ramp[index] = fitted;
            if !phase[index].is_nan() {
                output[index] = phase[index] - fitted;
            }
        }
    }
    Ok(DerampOutputs {
        phase: output,
        ramp,
    })
}

fn build_design(xy: &[f64], rows: impl Iterator<Item = usize>) -> Vec<f64> {
    let mut design = Vec::new();
    for row in rows {
        design.extend_from_slice(&[xy[row * 2] / 1000.0, xy[row * 2 + 1] / 1000.0, 1.0]);
    }
    design
}

fn validate_phase_shape(phase: &[f64], n_ps: usize, n_ifg: usize) -> Result<(), Stage7Error> {
    if n_ps == 0 || n_ifg == 0 || phase.len() != n_ps * n_ifg {
        return Err(Stage7Error::new(
            "phase must be a non-empty n_ps by n_ifg matrix",
        ));
    }
    Ok(())
}
