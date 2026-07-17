use super::Stage5Error;
use crate::stages::stage1::{Complex32, Matrix};

#[derive(Clone, Debug, PartialEq)]
pub struct Rc2Correction {
    pub phase_corrected: Matrix<Complex32>,
    pub phase_rereferenced: Option<Matrix<Complex32>>,
}

pub fn ifg_standard_deviation(
    phase: &Matrix<Complex32>,
    phase_patch: &Matrix<Complex32>,
    bperp: &Matrix<f64>,
    k_ps: &[f64],
    c_ps: &[f64],
) -> Result<Vec<f32>, Stage5Error> {
    if phase.rows == 0
        || phase.cols == 0
        || phase_patch.rows != phase.rows
        || phase_patch.cols != phase.cols
        || bperp.rows != phase.rows
        || bperp.cols != phase.cols
        || k_ps.len() != phase.rows
        || c_ps.len() != phase.rows
    {
        return Err(Stage5Error::InvalidInput(
            "IFG standard-deviation shapes do not match",
        ));
    }
    let mut sums = vec![0.0; phase.cols];
    for row in 0..phase.rows {
        for col in 0..phase.cols {
            let difference = wrap_phase(
                f64::from(phase.row(row)[col].arg())
                    - f64::from(phase_patch.row(row)[col].arg())
                    - k_ps[row] * bperp.row(row)[col]
                    - c_ps[row],
            );
            sums[col] += difference * difference;
        }
    }
    let degrees = 180.0 / std::f64::consts::PI;
    Ok(sums
        .into_iter()
        .map(|sum| ((sum / phase.rows as f64).sqrt() * degrees) as f32)
        .collect())
}

pub fn rc2_correction(
    phase: &Matrix<Complex32>,
    phase_patch: &Matrix<Complex32>,
    bperp: &Matrix<f64>,
    k_ps: &[f64],
    c_ps: &[f64],
    small_baseline: bool,
    master_ix: usize,
) -> Result<Rc2Correction, Stage5Error> {
    let expected_baseline_cols = if small_baseline {
        phase.cols
    } else {
        phase.cols.saturating_sub(1)
    };
    if phase.rows != bperp.rows
        || bperp.cols != expected_baseline_cols
        || k_ps.len() != phase.rows
        || c_ps.len() != phase.rows
        || (!small_baseline
            && (phase_patch.rows != phase.rows
                || phase_patch.cols != phase.cols.saturating_sub(1)
                || !(1..=phase.cols).contains(&master_ix)))
    {
        return Err(Stage5Error::InvalidInput(
            "RC2 correction shapes do not match",
        ));
    }
    let master = master_ix.saturating_sub(1);
    let mut corrected = Vec::with_capacity(phase.values.len());
    for row in 0..phase.rows {
        for col in 0..phase.cols {
            let baseline = if small_baseline {
                bperp.row(row)[col]
            } else if col < master {
                bperp.row(row)[col]
            } else if col == master {
                0.0
            } else {
                bperp.row(row)[col - 1]
            };
            let correction = if small_baseline {
                -k_ps[row] * baseline
            } else {
                -(k_ps[row] * baseline + c_ps[row])
            };
            corrected.push(
                phase.row(row)[col]
                    * Complex32::new(correction.cos() as f32, correction.sin() as f32),
            );
        }
    }
    let phase_rereferenced = if small_baseline {
        None
    } else {
        let mut values = Vec::with_capacity(phase.values.len());
        for row in 0..phase.rows {
            for col in 0..phase.cols {
                values.push(if col < master {
                    phase_patch.row(row)[col]
                } else if col == master {
                    Complex32::new(1.0, 0.0)
                } else {
                    phase_patch.row(row)[col - 1]
                });
            }
        }
        Some(Matrix {
            rows: phase.rows,
            cols: phase.cols,
            values,
        })
    };
    Ok(Rc2Correction {
        phase_corrected: Matrix {
            rows: phase.rows,
            cols: phase.cols,
            values: corrected,
        },
        phase_rereferenced,
    })
}

pub fn format_merged_rc2(values: &Matrix<Complex32>) -> Matrix<Complex32> {
    let mut transposed = vec![Complex32::new(0.0, 0.0); values.values.len()];
    for row in 0..values.rows {
        for col in 0..values.cols {
            let mut value = values.row(row)[col];
            let magnitude = value.norm();
            if magnitude != 0.0 {
                value /= magnitude;
            }
            transposed[col * values.rows + row] = value;
        }
    }
    Matrix {
        rows: values.cols,
        cols: values.rows,
        values: transposed,
    }
}

fn wrap_phase(value: f64) -> f64 {
    (value + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_master_rc2_inserts_unit_patch_phase_at_master() {
        let phase = Matrix::new(1, 3, vec![Complex32::new(1.0, 0.0); 3]).unwrap();
        let patch = Matrix::new(1, 2, vec![Complex32::new(0.0, 1.0); 2]).unwrap();
        let baseline = Matrix::new(1, 2, vec![-10.0, 10.0]).unwrap();
        let result = rc2_correction(&phase, &patch, &baseline, &[0.0], &[0.0], false, 2).unwrap();
        assert_eq!(
            result.phase_rereferenced.unwrap().row(0),
            &[
                Complex32::new(0.0, 1.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 1.0)
            ]
        );
    }
}
