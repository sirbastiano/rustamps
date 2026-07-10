use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use super::stage4_edge_stats_stats::{
    std_max_rows_real, variance_cols_complex, variance_cols_real,
};
use crate::build_pool;
use crate::weighted_fit_native::{
    weighted_affine_fit_rows, weighted_slope_fit_rows_complex, weighted_slope_fit_rows_real,
    wrap_phase,
};

pub(super) fn stage4_edge_stats_outputs(
    ph_slice: &[Complex64],
    n_node: usize,
    n_ifg: usize,
    edge_a: &[usize],
    edge_b: &[usize],
    bperp: &[f64],
    day: &[f64],
    time_win: f64,
    small_baseline: bool,
    threads: usize,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n_edge = edge_a.len();
    let mut ps_std = vec![f64::INFINITY; n_node];
    let mut ps_max = vec![f64::INFINITY; n_node];
    if n_edge == 0 || n_ifg == 0 {
        return Ok((ps_std, ps_max));
    }

    let pool = build_pool(threads)?;
    let mut dph_space = vec![Complex64::new(0.0, 0.0); n_edge * n_ifg];
    match &pool {
        Some(pool) => pool.install(|| {
            dph_space
                .par_chunks_mut(n_ifg)
                .enumerate()
                .for_each(|(edge_ix, row)| {
                    let a_ix = edge_a[edge_ix];
                    let b_ix = edge_b[edge_ix];
                    for ifg_ix in 0..n_ifg {
                        row[ifg_ix] = ph_slice[b_ix * n_ifg + ifg_ix]
                            * ph_slice[a_ix * n_ifg + ifg_ix].conj();
                    }
                });
        }),
        None => {
            for edge_ix in 0..n_edge {
                let a_ix = edge_a[edge_ix];
                let b_ix = edge_b[edge_ix];
                let row = &mut dph_space[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                for ifg_ix in 0..n_ifg {
                    row[ifg_ix] =
                        ph_slice[b_ix * n_ifg + ifg_ix] * ph_slice[a_ix * n_ifg + ifg_ix].conj();
                }
            }
        }
    }

    let (edge_std, edge_max) = if !small_baseline {
        if day.len() != n_ifg {
            return Err(PyValueError::new_err(
                "stage4_edge_stats day length must match phase width",
            ));
        }
        let time_win_f = time_win.max(1.0e-6);
        let mut time_diff_all = vec![0.0_f64; n_ifg * n_ifg];
        let mut weight_all = vec![0.0_f64; n_ifg * n_ifg];
        for row_ix in 0..n_ifg {
            let mut weight_sum = 0.0_f64;
            for col_ix in 0..n_ifg {
                let diff = day[row_ix] - day[col_ix];
                time_diff_all[row_ix * n_ifg + col_ix] = diff;
                let weight = (-(diff * diff) / (2.0 * time_win_f * time_win_f)).exp();
                weight_all[row_ix * n_ifg + col_ix] = weight;
                weight_sum += weight;
            }
            if weight_sum <= 0.0 {
                let fill = 1.0 / n_ifg as f64;
                for col_ix in 0..n_ifg {
                    weight_all[row_ix * n_ifg + col_ix] = fill;
                }
            } else {
                for col_ix in 0..n_ifg {
                    weight_all[row_ix * n_ifg + col_ix] /= weight_sum;
                }
            }
        }

        let mut dph_smooth0 = vec![Complex64::new(0.0, 0.0); n_edge * n_ifg];
        match &pool {
            Some(pool) => pool.install(|| {
                dph_smooth0
                    .par_chunks_mut(n_ifg)
                    .enumerate()
                    .for_each(|(edge_ix, row)| {
                        let source = &dph_space[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                        for out_ix in 0..n_ifg {
                            let weights = &weight_all[out_ix * n_ifg..(out_ix + 1) * n_ifg];
                            let mut accum = Complex64::new(0.0, 0.0);
                            for src_ix in 0..n_ifg {
                                accum += source[src_ix] * weights[src_ix];
                            }
                            row[out_ix] = accum;
                        }
                    });
            }),
            None => {
                for edge_ix in 0..n_edge {
                    let source = &dph_space[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                    let row = &mut dph_smooth0[edge_ix * n_ifg..(edge_ix + 1) * n_ifg];
                    for out_ix in 0..n_ifg {
                        let weights = &weight_all[out_ix * n_ifg..(out_ix + 1) * n_ifg];
                        let mut accum = Complex64::new(0.0, 0.0);
                        for src_ix in 0..n_ifg {
                            accum += source[src_ix] * weights[src_ix];
                        }
                        row[out_ix] = accum;
                    }
                }
            }
        }

        let mut dph_smooth2 = dph_smooth0.clone();
        for edge_ix in 0..n_edge {
            for ifg_ix in 0..n_ifg {
                let diag = weight_all[ifg_ix * n_ifg + ifg_ix];
                dph_smooth2[edge_ix * n_ifg + ifg_ix] -= dph_space[edge_ix * n_ifg + ifg_ix] * diag;
            }
        }

        let columns = match &pool {
            Some(pool) => pool.install(|| {
                (0..n_ifg)
                    .into_par_iter()
                    .map(|ifg_ix| {
                        let time_diff = &time_diff_all[ifg_ix * n_ifg..(ifg_ix + 1) * n_ifg];
                        let weight = &weight_all[ifg_ix * n_ifg..(ifg_ix + 1) * n_ifg];
                        let mut dph_mean_adj = vec![0.0_f64; n_edge * n_ifg];
                        let mut dph_mean = vec![Complex64::new(0.0, 0.0); n_edge];
                        for edge_ix in 0..n_edge {
                            let mean = dph_smooth0[edge_ix * n_ifg + ifg_ix];
                            dph_mean[edge_ix] = mean;
                            let mean_conj = mean.conj();
                            for col_ix in 0..n_ifg {
                                dph_mean_adj[edge_ix * n_ifg + col_ix] =
                                    (dph_space[edge_ix * n_ifg + col_ix] * mean_conj).arg();
                            }
                        }
                        let (m0, m1) = weighted_affine_fit_rows(
                            time_diff,
                            &dph_mean_adj,
                            n_edge,
                            n_ifg,
                            weight,
                        );
                        let mut dph_mean_adj2 = vec![0.0_f64; n_edge * n_ifg];
                        for edge_ix in 0..n_edge {
                            for col_ix in 0..n_ifg {
                                let detrended = dph_mean_adj[edge_ix * n_ifg + col_ix]
                                    - (m0[edge_ix] + m1[edge_ix] * time_diff[col_ix]);
                                dph_mean_adj2[edge_ix * n_ifg + col_ix] = wrap_phase(detrended);
                            }
                        }
                        let (m20, _) = weighted_affine_fit_rows(
                            time_diff,
                            &dph_mean_adj2,
                            n_edge,
                            n_ifg,
                            weight,
                        );
                        let mut column = vec![Complex64::new(0.0, 0.0); n_edge];
                        for edge_ix in 0..n_edge {
                            column[edge_ix] = dph_mean[edge_ix]
                                * Complex64::from_polar(1.0, m0[edge_ix] + m20[edge_ix]);
                        }
                        column
                    })
                    .collect::<Vec<_>>()
            }),
            None => {
                let mut cols = Vec::with_capacity(n_ifg);
                for ifg_ix in 0..n_ifg {
                    let time_diff = &time_diff_all[ifg_ix * n_ifg..(ifg_ix + 1) * n_ifg];
                    let weight = &weight_all[ifg_ix * n_ifg..(ifg_ix + 1) * n_ifg];
                    let mut dph_mean_adj = vec![0.0_f64; n_edge * n_ifg];
                    let mut dph_mean = vec![Complex64::new(0.0, 0.0); n_edge];
                    for edge_ix in 0..n_edge {
                        let mean = dph_smooth0[edge_ix * n_ifg + ifg_ix];
                        dph_mean[edge_ix] = mean;
                        let mean_conj = mean.conj();
                        for col_ix in 0..n_ifg {
                            dph_mean_adj[edge_ix * n_ifg + col_ix] =
                                (dph_space[edge_ix * n_ifg + col_ix] * mean_conj).arg();
                        }
                    }
                    let (m0, m1) =
                        weighted_affine_fit_rows(time_diff, &dph_mean_adj, n_edge, n_ifg, weight);
                    let mut dph_mean_adj2 = vec![0.0_f64; n_edge * n_ifg];
                    for edge_ix in 0..n_edge {
                        for col_ix in 0..n_ifg {
                            let detrended = dph_mean_adj[edge_ix * n_ifg + col_ix]
                                - (m0[edge_ix] + m1[edge_ix] * time_diff[col_ix]);
                            dph_mean_adj2[edge_ix * n_ifg + col_ix] = wrap_phase(detrended);
                        }
                    }
                    let (m20, _) =
                        weighted_affine_fit_rows(time_diff, &dph_mean_adj2, n_edge, n_ifg, weight);
                    let mut column = vec![Complex64::new(0.0, 0.0); n_edge];
                    for edge_ix in 0..n_edge {
                        column[edge_ix] = dph_mean[edge_ix]
                            * Complex64::from_polar(1.0, m0[edge_ix] + m20[edge_ix]);
                    }
                    cols.push(column);
                }
                cols
            }
        };

        let mut dph_smooth = dph_smooth0;
        for ifg_ix in 0..n_ifg {
            for edge_ix in 0..n_edge {
                dph_smooth[edge_ix * n_ifg + ifg_ix] = columns[ifg_ix][edge_ix];
            }
        }

        let mut dph_noise = vec![0.0_f64; n_edge * n_ifg];
        let mut dph_noise2 = vec![0.0_f64; n_edge * n_ifg];
        for edge_ix in 0..n_edge {
            for ifg_ix in 0..n_ifg {
                dph_noise[edge_ix * n_ifg + ifg_ix] = (dph_space[edge_ix * n_ifg + ifg_ix]
                    * dph_smooth[edge_ix * n_ifg + ifg_ix].conj())
                .arg();
                dph_noise2[edge_ix * n_ifg + ifg_ix] = (dph_space[edge_ix * n_ifg + ifg_ix]
                    * dph_smooth2[edge_ix * n_ifg + ifg_ix].conj())
                .arg();
            }
        }

        let ddof_var = if n_edge > 1 { 1 } else { 0 };
        let ifg_var = variance_cols_real(&dph_noise2, n_edge, n_ifg, ddof_var);
        let w_ifg: Vec<f64> = ifg_var
            .iter()
            .map(|&value| {
                if value == 0.0 {
                    f64::INFINITY
                } else {
                    1.0 / value
                }
            })
            .collect();
        let k_edge = weighted_slope_fit_rows_real(bperp, &dph_noise, n_edge, n_ifg, &w_ifg);
        for edge_ix in 0..n_edge {
            let slope = k_edge[edge_ix];
            for ifg_ix in 0..n_ifg {
                dph_noise[edge_ix * n_ifg + ifg_ix] -= slope * bperp[ifg_ix];
            }
        }
        let ddof = if n_ifg > 1 { 1 } else { 0 };
        std_max_rows_real(&dph_noise, n_edge, n_ifg, ddof)
    } else {
        let ddof_var = if n_edge > 1 { 1 } else { 0 };
        let ifg_var = variance_cols_complex(&dph_space, n_edge, n_ifg, ddof_var);
        let w_ifg: Vec<f64> = ifg_var
            .iter()
            .map(|&value| {
                if value == 0.0 {
                    f64::INFINITY
                } else {
                    1.0 / value
                }
            })
            .collect();
        let k_edge = weighted_slope_fit_rows_complex(bperp, &dph_space, n_edge, n_ifg, &w_ifg);
        let mut ang = vec![0.0_f64; n_edge * n_ifg];
        for edge_ix in 0..n_edge {
            let slope = k_edge[edge_ix];
            for ifg_ix in 0..n_ifg {
                ang[edge_ix * n_ifg + ifg_ix] =
                    (dph_space[edge_ix * n_ifg + ifg_ix] - slope * bperp[ifg_ix]).arg();
            }
        }
        let ddof = if n_ifg > 1 { 1 } else { 0 };
        std_max_rows_real(&ang, n_edge, n_ifg, ddof)
    };

    for edge_ix in 0..n_edge {
        let a_ix = edge_a[edge_ix];
        let b_ix = edge_b[edge_ix];
        ps_std[a_ix] = ps_std[a_ix].min(edge_std[edge_ix]);
        ps_std[b_ix] = ps_std[b_ix].min(edge_std[edge_ix]);
        ps_max[a_ix] = ps_max[a_ix].min(edge_max[edge_ix]);
        ps_max[b_ix] = ps_max[b_ix].min(edge_max[edge_ix]);
    }
    Ok((ps_std, ps_max))
}
