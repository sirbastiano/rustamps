use std::path::Path;

use num_complex::Complex32;
use pystamps_core::stage6::unwrap_ifg_sets;
use pystamps_io::{read_mat, MatFile};

use crate::RunConfig;

use super::super::mat::{complex32, numeric_f64, shape};
use super::super::params::Params;
use super::fingerprint;
use super::wrapped;

pub struct Options {
    pub grid_size: f64,
    pub prefilter: bool,
    pub filter_window: usize,
    pub filter_alpha: f64,
    pub time_window: f64,
    pub trial_wraps: f64,
    pub parallel: bool,
    pub custom_pool: bool,
    pub ifg_workers: usize,
    pub max_flow_passes: Option<usize>,
}

pub struct Input {
    pub fingerprint: u64,
    pub n_ps: usize,
    pub n_ifg: usize,
    pub master: usize,
    pub unwrap: Vec<usize>,
    pub phase: Vec<Complex32>,
    pub phase_restore: Vec<f32>,
    pub xy: Vec<[f64; 2]>,
    pub day: Vec<f64>,
    pub bperp: Vec<f64>,
    pub options: Options,
}

pub fn load(root: &Path, params: &Params, config: &RunConfig) -> Result<Input, String> {
    validate_flags(params)?;
    let ps = required(root, "ps2.mat")?;
    let ph_file = required(root, "ph2.mat")?;
    let pm = required(root, "pm2.mat")?;
    let phase_shape = shape(&ph_file, "ph")?;
    if phase_shape.len() != 2 || phase_shape[0] == 0 || phase_shape[1] < 2 {
        return Err(format!(
            "ph2.ph must have shape (n_ps, n_ifg>=2), found {phase_shape:?}"
        ));
    }
    let (n_ps, n_ifg) = (phase_shape[0], phase_shape[1]);
    if integer(&ps, "n_ps")? != n_ps {
        return Err("ps2.n_ps does not match ph2.ph rows".to_owned());
    }
    let master_one = integer(&ps, "master_ix")?;
    if !(1..=n_ifg).contains(&master_one) {
        return Err("ps2.master_ix is outside ph2.ph columns".to_owned());
    }
    let master = master_one - 1;
    let dropped = params.indices("drop_ifg_index")?;
    let sets =
        unwrap_ifg_sets(n_ifg, master, &dropped, false).map_err(|error| error.to_string())?;
    if sets.solve_indices.is_empty() {
        return Err("no non-master interferograms remain for Stage 6".to_owned());
    }
    let day = vector(&ps, "day", n_ifg)?;
    let bperp = vector(&ps, "bperp", n_ifg)?;
    let xy_values = numeric_f64(&ps, "xy")?;
    if shape(&ps, "xy")? != [n_ps, 3] || xy_values.len() != n_ps * 3 {
        return Err("ps2.xy must have shape (n_ps, 3)".to_owned());
    }
    let xy = xy_values
        .chunks_exact(3)
        .map(|row| [row[1], row[2]])
        .collect::<Vec<_>>();
    let ph2 = complex32(&ph_file, "ph")?;
    let (phase, phase_restore) = wrapped::build(root, params, &ps, &pm, &ph2, n_ps, n_ifg, master)?;
    let max_topo_error = params.scalar("max_topo_err", 20.0)?;
    let wavelength = params.scalar("lambda", f64::NAN)?;
    let mean_range = scalar(&ps, "mean_range", 830_000.0)?;
    let incidence = scalar(&ps, "mean_incidence", 23_f64.to_radians())?;
    let scale = wavelength * mean_range * incidence.sin() / (4.0 * std::f64::consts::PI);
    if !scale.is_finite() || scale <= 0.0 || !max_topo_error.is_finite() {
        return Err("invalid wavelength/range/incidence/max_topo_err geometry".to_owned());
    }
    let baseline_span = bperp.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - bperp.iter().copied().fold(f64::INFINITY, f64::min);
    let trial_wraps = baseline_span * (max_topo_error / scale) / std::f64::consts::TAU;
    let mut grid_size = params.scalar("unwrap_grid_size", 200.0)?;
    if grid_size <= 0.0 {
        grid_size = 20.0;
    }
    grid_size *= config.runtime.stage6_grid_scale;
    let mut filter_window = params.scalar("unwrap_gold_n_win", 32.0)?.round() as usize;
    if filter_window == 0 {
        filter_window = 32;
    }
    if filter_window % 2 != 0 {
        return Err("unwrap_gold_n_win must be even".to_owned());
    }
    let filter_alpha = params.scalar("unwrap_gold_alpha", 0.8)?;
    let time_window = params.scalar("unwrap_time_win", 730.0)?;
    if [grid_size, filter_alpha, time_window, trial_wraps]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err("Stage 6 numeric parameters must be finite".to_owned());
    }
    let mut input = Input {
        fingerprint: 0,
        n_ps,
        n_ifg,
        master,
        unwrap: sets.solve_indices,
        phase,
        phase_restore,
        xy,
        day,
        bperp,
        options: Options {
            grid_size,
            prefilter: params.flag("unwrap_prefilter_flag", true)?,
            filter_window,
            filter_alpha,
            time_window,
            trial_wraps,
            parallel: config.runtime.cpu_workers != 1,
            custom_pool: config.runtime.cpu_workers > 0,
            ifg_workers: config.runtime.stage6_ifg_workers,
            max_flow_passes: (config.runtime.stage6_max_flow_passes > 0)
                .then_some(config.runtime.stage6_max_flow_passes),
        },
    };
    input.fingerprint = fingerprint::input(&input);
    Ok(input)
}

fn validate_flags(params: &Params) -> Result<(), String> {
    for (name, default) in [
        ("small_baseline_flag", false),
        ("subtr_tropo", false),
        ("unwrap_hold_good_values", false),
        ("unwrap_patch_phase", false),
    ] {
        if params.flag(name, default)? {
            return Err(format!("{name}='y' is unsupported by native Stage 6"));
        }
    }
    let method = params.text("unwrap_method", "3D")?.to_ascii_uppercase();
    if !matches!(method.as_str(), "3D" | "3D_NEW" | "3D_FULL") {
        return Err(format!(
            "unwrap_method={method} is unsupported; use 3D_FULL"
        ));
    }
    if !params.flag("unwrap_la_error_flag", true)? {
        return Err("unwrap_la_error_flag='n' is unsupported by the 3D_FULL path".to_owned());
    }
    if params.flag("unwrap_spatial_cost_func_flag", false)? {
        return Err("unwrap_spatial_cost_func_flag='y' is unsupported".to_owned());
    }
    Ok(())
}

fn required(root: &Path, name: &str) -> Result<MatFile, String> {
    let path = root.join(name);
    if !path.is_file() {
        return Err(format!("missing required Stage 6 artifact {name}"));
    }
    read_mat(path).map_err(|error| error.to_string())
}

fn integer(file: &MatFile, key: &str) -> Result<usize, String> {
    let value = numeric_f64(file, key)?
        .first()
        .copied()
        .ok_or_else(|| format!("{key} is empty"))?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        Err(format!("{key} is not a non-negative integer"))
    } else {
        Ok(value as usize)
    }
}

fn scalar(file: &MatFile, key: &str, default: f64) -> Result<f64, String> {
    if file.contains_key(key) {
        numeric_f64(file, key)?
            .first()
            .copied()
            .ok_or_else(|| format!("{key} is empty"))
    } else {
        Ok(default)
    }
}

fn vector(file: &MatFile, key: &str, expected: usize) -> Result<Vec<f64>, String> {
    let values = numeric_f64(file, key)?;
    if values.len() != expected {
        Err(format!(
            "ps2.{key} has {} values; expected {expected}",
            values.len()
        ))
    } else if values.iter().any(|value| !value.is_finite()) {
        Err(format!("ps2.{key} contains non-finite values"))
    } else {
        Ok(values)
    }
}
