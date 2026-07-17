use super::types::{
    Stage2Config, Stage2Error, Stage2Input, Stage2Kernel, Stage2Output, Stage2State,
};

fn validate(input: &Stage2Input, config: &Stage2Config) -> Result<(), Stage2Error> {
    let n_ps = input.phase.rows;
    if n_ps == 0 || input.phase.cols == 0 {
        return Err(Stage2Error::InvalidInput("phase must be non-empty"));
    }
    if input.bperp_mat.rows != n_ps {
        return Err(Stage2Error::InvalidInput(
            "baseline rows must match phase rows",
        ));
    }
    let expected_cols = if input.small_baseline {
        input.phase.cols
    } else {
        input.phase.cols.saturating_sub(1)
    };
    if input.bperp_mat.cols != expected_cols {
        return Err(Stage2Error::InvalidInput(
            "baseline columns do not match processing mode",
        ));
    }
    if input.xy.rows != n_ps || input.xy.cols != 3 {
        return Err(Stage2Error::InvalidInput("xy must be n_ps by 3"));
    }
    if input.amplitude_dispersion.len() != n_ps {
        return Err(Stage2Error::InvalidInput(
            "D_A length must match phase rows",
        ));
    }
    if !input.small_baseline && !(1..=input.phase.cols).contains(&input.master_ix) {
        return Err(Stage2Error::InvalidInput(
            "master_ix must be valid and one-based",
        ));
    }
    if config.max_iterations == 0 || !config.convergence.is_finite() || config.convergence < 0.0 {
        return Err(Stage2Error::InvalidInput(
            "invalid convergence configuration",
        ));
    }
    Ok(())
}

pub fn valid_all_ifg_rows(
    phase: &crate::stages::stage1::Matrix<crate::stages::stage1::Complex32>,
) -> Vec<bool> {
    (0..phase.rows)
        .map(|row| {
            phase
                .row(row)
                .iter()
                .all(|value| value.re != 0.0 || value.im != 0.0)
        })
        .collect()
}

pub fn run_stage2<K: Stage2Kernel>(
    input: &Stage2Input,
    config: &Stage2Config,
    kernel: &mut K,
) -> Result<Stage2Output, Stage2Error> {
    validate(input, config)?;
    let n_ps = input.phase.rows;
    let mut state = Stage2State {
        iteration: 1,
        k_ps: vec![0.0; n_ps],
        c_ps: vec![0.0; n_ps],
        coherence: vec![0.0; n_ps],
        previous_coherence: vec![0.0; n_ps],
        weighting: input
            .amplitude_dispersion
            .iter()
            .map(|&value| if value == 0.0 { 0.0 } else { 1.0 / value })
            .collect(),
        previous_rms_change: 0.0,
    };
    loop {
        let iteration = kernel.estimate(input, &state)?;
        if iteration.k_ps.len() != n_ps
            || iteration.c_ps.len() != n_ps
            || iteration.coherence.len() != n_ps
            || iteration.weighting.len() != n_ps
        {
            return Err(Stage2Error::InvalidInput(
                "kernel output row count mismatch",
            ));
        }
        let rms = (iteration
            .coherence
            .iter()
            .zip(&state.previous_coherence)
            .map(|(current, previous)| (current - previous).powi(2))
            .sum::<f64>()
            / n_ps as f64)
            .sqrt();
        let change = rms - state.previous_rms_change;
        let stop = change.abs() < config.convergence || state.iteration >= config.max_iterations;
        if stop {
            return Ok(Stage2Output {
                iterations: state.iteration,
                k_ps: iteration.k_ps,
                filter_k_ps: state.k_ps,
                c_ps: iteration.c_ps,
                coherence: iteration.coherence,
                weighting: state.weighting,
                phase_residual: iteration.phase_residual,
                phase_patch: iteration.phase_patch,
                gamma_rms: rms,
                gamma_change: change,
            });
        }
        state.previous_coherence = iteration.coherence.clone();
        state.previous_rms_change = rms;
        state.k_ps = iteration.k_ps;
        state.c_ps = iteration.c_ps;
        state.coherence = iteration.coherence;
        state.weighting = iteration.weighting;
        state.iteration += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::Stage2Iteration;
    use super::*;
    use crate::stages::stage1::{Complex32, Matrix};

    struct FixedKernel;

    impl Stage2Kernel for FixedKernel {
        fn estimate(
            &mut self,
            input: &Stage2Input,
            state: &Stage2State,
        ) -> Result<Stage2Iteration, Stage2Error> {
            let value = if state.iteration == 1 { 0.5 } else { 0.5001 };
            Ok(Stage2Iteration {
                k_ps: vec![0.0; input.phase.rows],
                c_ps: vec![0.0; input.phase.rows],
                coherence: vec![value; input.phase.rows],
                weighting: vec![1.0; input.phase.rows],
                phase_residual: Matrix::new(input.phase.rows, 1, vec![0.0; input.phase.rows])
                    .unwrap(),
                phase_patch: Matrix::new(
                    input.phase.rows,
                    1,
                    vec![Complex32::new(1.0, 0.0); input.phase.rows],
                )
                .unwrap(),
            })
        }
    }

    #[test]
    fn state_machine_stops_on_change_in_rms() {
        let input = Stage2Input {
            phase: Matrix::new(1, 2, vec![Complex32::new(1.0, 0.0); 2]).unwrap(),
            bperp_mat: Matrix::new(1, 1, vec![10.0]).unwrap(),
            xy: Matrix::new(1, 3, vec![1.0, 0.0, 0.0]).unwrap(),
            amplitude_dispersion: vec![0.2],
            master_ix: 1,
            small_baseline: false,
        };
        let output = run_stage2(&input, &Stage2Config::default(), &mut FixedKernel).unwrap();
        assert_eq!(output.iterations, 3);
    }

    struct InitialWeightKernel;

    impl Stage2Kernel for InitialWeightKernel {
        fn estimate(
            &mut self,
            input: &Stage2Input,
            state: &Stage2State,
        ) -> Result<Stage2Iteration, Stage2Error> {
            if state.iteration == 1 {
                assert_eq!(state.weighting, vec![5.0]);
            }
            FixedKernel.estimate(input, state)
        }
    }

    #[test]
    fn first_iteration_uses_inverse_amplitude_dispersion_weighting() {
        let input = Stage2Input {
            phase: Matrix::new(1, 2, vec![Complex32::new(1.0, 0.0); 2]).unwrap(),
            bperp_mat: Matrix::new(1, 1, vec![10.0]).unwrap(),
            xy: Matrix::new(1, 3, vec![1.0, 0.0, 0.0]).unwrap(),
            amplitude_dispersion: vec![0.2],
            master_ix: 1,
            small_baseline: false,
        };
        run_stage2(&input, &Stage2Config::default(), &mut InitialWeightKernel).unwrap();
    }

    struct ChangedOutputKernel;

    impl Stage2Kernel for ChangedOutputKernel {
        fn estimate(
            &mut self,
            input: &Stage2Input,
            state: &Stage2State,
        ) -> Result<Stage2Iteration, Stage2Error> {
            Ok(Stage2Iteration {
                k_ps: vec![state.iteration as f64; input.phase.rows],
                c_ps: vec![0.0; input.phase.rows],
                coherence: vec![0.5; input.phase.rows],
                weighting: vec![99.0; input.phase.rows],
                phase_residual: Matrix::new(input.phase.rows, 1, vec![0.0; input.phase.rows])
                    .unwrap(),
                phase_patch: Matrix::new(
                    input.phase.rows,
                    1,
                    vec![Complex32::new(1.0, 0.0); input.phase.rows],
                )
                .unwrap(),
            })
        }
    }

    #[test]
    fn stopped_output_retains_the_state_used_for_the_final_grid() {
        let input = Stage2Input {
            phase: Matrix::new(1, 2, vec![Complex32::new(1.0, 0.0); 2]).unwrap(),
            bperp_mat: Matrix::new(1, 1, vec![10.0]).unwrap(),
            xy: Matrix::new(1, 3, vec![1.0, 0.0, 0.0]).unwrap(),
            amplitude_dispersion: vec![0.2],
            master_ix: 1,
            small_baseline: false,
        };
        let config = Stage2Config {
            max_iterations: 1,
            ..Stage2Config::default()
        };
        let output = run_stage2(&input, &config, &mut ChangedOutputKernel).unwrap();
        assert_eq!(output.k_ps, vec![1.0]);
        assert_eq!(output.filter_k_ps, vec![0.0]);
        assert_eq!(output.weighting, vec![5.0]);
        assert_eq!(output.gamma_rms, 0.5);
        assert_eq!(output.gamma_change, 0.5);
    }
}
