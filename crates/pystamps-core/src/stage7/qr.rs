//! Small shared-design least-squares solver using pivoted, reorthogonalized QR.

#[derive(Clone, Debug)]
pub(crate) struct QrSolver {
    n_obs: usize,
    n_cols: usize,
    q_columns: Vec<f64>,
    r: Vec<f64>,
    permutation: Vec<usize>,
}

impl QrSolver {
    pub(crate) fn factor(design: &[f64], n_obs: usize, n_cols: usize) -> Result<Self, String> {
        if n_obs == 0 || n_cols == 0 || n_obs < n_cols {
            return Err("least-squares design must have rows >= non-zero columns".into());
        }
        if design.len() != n_obs * n_cols {
            return Err("least-squares design shape does not match its data".into());
        }
        if design.iter().any(|value| !value.is_finite()) {
            return Err("least-squares design contains a non-finite value".into());
        }

        let mut columns = (0..n_cols)
            .map(|column| {
                (0..n_obs)
                    .map(|row| design[row * n_cols + column])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut norms = columns
            .iter()
            .map(|column| dot(column, column).sqrt())
            .collect::<Vec<_>>();
        let max_norm = norms.iter().copied().fold(0.0_f64, f64::max);
        let tolerance = max_norm * f64::EPSILON * (n_obs.max(n_cols) as f64) * 32.0;
        let mut permutation = (0..n_cols).collect::<Vec<_>>();
        let mut q_columns = vec![0.0; n_obs * n_cols];
        let mut r = vec![0.0; n_cols * n_cols];

        for step in 0..n_cols {
            let pivot = (step..n_cols)
                .max_by(|left, right| norms[*left].total_cmp(&norms[*right]))
                .expect("non-empty pivot range");
            if pivot != step {
                columns.swap(step, pivot);
                norms.swap(step, pivot);
                permutation.swap(step, pivot);
                for row in 0..step {
                    r.swap(row * n_cols + step, row * n_cols + pivot);
                }
            }

            let diagonal = dot(&columns[step], &columns[step]).sqrt();
            if !diagonal.is_finite() || diagonal <= tolerance {
                return Err(format!(
                    "least-squares design is rank deficient at column {step}"
                ));
            }
            r[step * n_cols + step] = diagonal;
            for row in 0..n_obs {
                q_columns[step * n_obs + row] = columns[step][row] / diagonal;
            }

            for column in (step + 1)..n_cols {
                let q = &q_columns[step * n_obs..(step + 1) * n_obs];
                let mut projection = dot(q, &columns[column]);
                axpy(&mut columns[column], q, -projection);
                let correction = dot(q, &columns[column]);
                axpy(&mut columns[column], q, -correction);
                projection += correction;
                r[step * n_cols + column] = projection;
                norms[column] = dot(&columns[column], &columns[column]).sqrt();
            }
        }

        Ok(Self {
            n_obs,
            n_cols,
            q_columns,
            r,
            permutation,
        })
    }

    pub(crate) fn solve_with(&self, mut value: impl FnMut(usize) -> f64) -> Vec<f64> {
        let mut projected = vec![0.0; self.n_cols];
        for (column, projected_value) in projected.iter_mut().enumerate() {
            let q = &self.q_columns[column * self.n_obs..(column + 1) * self.n_obs];
            *projected_value = q
                .iter()
                .enumerate()
                .map(|(row, coefficient)| coefficient * value(row))
                .sum();
        }

        let mut pivoted = vec![0.0; self.n_cols];
        for row in (0..self.n_cols).rev() {
            let trailing: f64 = ((row + 1)..self.n_cols)
                .map(|column| self.r[row * self.n_cols + column] * pivoted[column])
                .sum();
            pivoted[row] = (projected[row] - trailing) / self.r[row * self.n_cols + row];
        }
        let mut output = vec![0.0; self.n_cols];
        for column in 0..self.n_cols {
            output[self.permutation[column]] = pivoted[column];
        }
        output
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn axpy(target: &mut [f64], source: &[f64], scale: f64) {
    for (target_value, source_value) in target.iter_mut().zip(source) {
        *target_value += scale * source_value;
    }
}
