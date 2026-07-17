use rayon::prelude::*;

use super::qr::QrSolver;
use super::{Stage7Error, Stage7Inputs, Stage7Outputs};

pub fn estimate_scla(input: &Stage7Inputs<'_>) -> Result<Stage7Outputs, Stage7Error> {
    validate(input)?;
    let n_ps = input.n_ps;
    let n_ifg = input.n_ifg;
    let sequential_observations = input.unwrap_indices.len() - 1;
    let coestimate_velocity = input.unwrap_indices.len() >= 4;
    let day_difference = input
        .unwrap_indices
        .windows(2)
        .map(|pair| input.day[pair[1]] - input.day[pair[0]])
        .collect::<Vec<_>>();
    // A constant acquisition interval makes the velocity column a scaled
    // duplicate of the intercept.  StaMPS' least-squares solve tolerates that
    // rank deficiency; omitting only the redundant column preserves K exactly.
    let fit_sequential_velocity = coestimate_velocity && !is_effectively_constant(&day_difference);

    let mut mean_baseline_difference = vec![0.0; sequential_observations];
    for (observation, pair) in input.unwrap_indices.windows(2).enumerate() {
        mean_baseline_difference[observation] = (0..n_ps)
            .map(|row| {
                let offset = row * n_ifg;
                input.bperp_mat[offset + pair[1]] - input.bperp_mat[offset + pair[0]]
            })
            .sum::<f64>()
            / n_ps as f64;
    }

    let sequential_columns = if fit_sequential_velocity { 3 } else { 2 };
    let mut sequential_design = vec![0.0; sequential_observations * sequential_columns];
    for observation in 0..sequential_observations {
        let offset = observation * sequential_columns;
        sequential_design[offset] = 1.0;
        sequential_design[offset + 1] = mean_baseline_difference[observation];
        if fit_sequential_velocity {
            sequential_design[offset + 2] = day_difference[observation];
        }
    }
    let sequential_solver = QrSolver::factor(
        &sequential_design,
        sequential_observations,
        sequential_columns,
    )
    .map_err(Stage7Error::new)?;

    let mut solve_scales = Vec::new();
    let constant_solver = if coestimate_velocity {
        solve_scales = input
            .solve_indices
            .iter()
            .map(|&index| {
                let standard_deviation = input.ifg_std[index];
                if standard_deviation > 0.0 {
                    standard_deviation * std::f64::consts::PI / 180.0
                } else {
                    1.0
                }
            })
            .collect::<Vec<_>>();
        let mut design = vec![0.0; input.solve_indices.len() * 2];
        for (observation, &index) in input.solve_indices.iter().enumerate() {
            design[observation * 2] = 1.0 / solve_scales[observation];
            design[observation * 2 + 1] =
                (input.day[index] - input.day[input.master_index]) / solve_scales[observation];
        }
        Some(QrSolver::factor(&design, input.solve_indices.len(), 2).map_err(Stage7Error::new)?)
    } else {
        None
    };

    let mut k_ps_uw = vec![0.0_f64; n_ps];
    let mut c_ps_uw = vec![0.0_f32; n_ps];
    let mut ph_scla = vec![0.0_f32; n_ps * n_ifg];
    ph_scla
        .par_chunks_mut(n_ifg)
        .zip(k_ps_uw.par_iter_mut())
        .zip(c_ps_uw.par_iter_mut())
        .enumerate()
        .for_each(|(row, ((phase_output, k_output), c_output))| {
            let row_offset = row * n_ifg;
            let coefficients = sequential_solver.solve_with(|observation| {
                let pair = &input.unwrap_indices[observation..observation + 2];
                input.ph_proc[row_offset + pair[1]] - input.ph_proc[row_offset + pair[0]]
            });
            let k = coefficients[1];
            *k_output = k;
            for (column, phase_value) in phase_output.iter_mut().enumerate() {
                *phase_value = (k * input.bperp_mat[row_offset + column]) as f32;
            }

            *c_output = if let Some(solver) = constant_solver.as_ref() {
                solver.solve_with(|observation| {
                    let column = input.solve_indices[observation];
                    (input.ph_proc[row_offset + column] - phase_output[column] as f64)
                        / solve_scales[observation]
                })[0] as f32
            } else {
                let sum: f64 = input
                    .solve_indices
                    .iter()
                    .map(|&column| input.ph_proc[row_offset + column] - phase_output[column] as f64)
                    .sum();
                (sum / input.solve_indices.len() as f64) as f32
            };
        });
    if k_ps_uw.iter().any(|value| value.is_infinite())
        || c_ps_uw.iter().any(|value| value.is_infinite())
        || ph_scla.iter().any(|value| value.is_infinite())
    {
        return Err(Stage7Error::new(
            "Stage 7 least-squares output overflowed its numeric contract",
        ));
    }

    let mut ifg_vcm = vec![0.0; n_ifg * n_ifg];
    for (index, standard_deviation) in input.ifg_std.iter().enumerate() {
        let radians = standard_deviation * std::f64::consts::PI / 180.0;
        ifg_vcm[index * n_ifg + index] = radians * radians;
    }
    Ok(Stage7Outputs {
        k_ps_uw,
        c_ps_uw,
        ph_scla,
        ifg_vcm,
    })
}

fn is_effectively_constant(values: &[f64]) -> bool {
    let first = values[0];
    let scale = values.iter().fold(first.abs().max(1.0), |current, value| {
        current.max(value.abs())
    });
    let tolerance = scale * f64::EPSILON * values.len() as f64 * 32.0;
    values
        .iter()
        .all(|value| (*value - first).abs() <= tolerance)
}

fn validate(input: &Stage7Inputs<'_>) -> Result<(), Stage7Error> {
    let matrix_len = input
        .n_ps
        .checked_mul(input.n_ifg)
        .ok_or_else(|| Stage7Error::new("Stage 7 matrix shape overflows usize"))?;
    if input.n_ps == 0 || input.n_ifg == 0 {
        return Err(Stage7Error::new("Stage 7 matrices must be non-empty"));
    }
    if input.ph_proc.len() != matrix_len || input.bperp_mat.len() != matrix_len {
        return Err(Stage7Error::new(
            "Stage 7 matrix data does not match n_ps by n_ifg",
        ));
    }
    if input.day.len() != input.n_ifg || input.ifg_std.len() != input.n_ifg {
        return Err(Stage7Error::new("day and ifg_std must match n_ifg"));
    }
    if input.master_index >= input.n_ifg {
        return Err(Stage7Error::new(
            "master_index is outside the interferogram stack",
        ));
    }
    validate_indices(input.unwrap_indices, input.n_ifg, 2, true, "unwrap_indices")?;
    validate_indices(input.solve_indices, input.n_ifg, 2, false, "solve_indices")?;
    if input.bperp_mat.iter().any(|value| !value.is_finite())
        || input.day.iter().any(|value| !value.is_finite())
        || input
            .ifg_std
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || input.ph_proc.iter().any(|value| value.is_infinite())
    {
        return Err(Stage7Error::new(
            "Stage 7 input contains an unsupported non-finite value",
        ));
    }
    Ok(())
}

fn validate_indices(
    indices: &[usize],
    upper: usize,
    minimum: usize,
    require_sorted: bool,
    name: &str,
) -> Result<(), Stage7Error> {
    if indices.len() < minimum || indices.iter().any(|&index| index >= upper) {
        return Err(Stage7Error::new(format!(
            "{name} is too short or outside n_ifg"
        )));
    }
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != indices.len() || (require_sorted && sorted != indices) {
        return Err(Stage7Error::new(format!(
            "{name} must contain unique sorted indices"
        )));
    }
    Ok(())
}
