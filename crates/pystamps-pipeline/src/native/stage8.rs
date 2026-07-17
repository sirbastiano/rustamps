use std::collections::BTreeMap;
use std::path::Path;

use pystamps_core::stage8::{estimate_scn, ScnConfig, ScnInputs};
use pystamps_io::{read_mat, write_mat, MatFile, StageTransaction};

use crate::{PipelineError, RunConfig};

use super::failure;
use super::mat::{f32_array, f64_array, numeric_f32, numeric_f64, shape};
use super::params::Params;

pub fn run(root: &Path, _config: &RunConfig) -> Result<String, PipelineError> {
    let params = Params::load(root).map_err(|error| failure(8, error))?;
    if params
        .flag("scn_kriging_flag", false)
        .map_err(|error| failure(8, error))?
    {
        return Err(failure(
            8,
            "scn_kriging_flag='y' is unsupported; refusing a Gaussian substitution",
        ));
    }
    let loaded = load(root).map_err(|error| failure(8, error))?;
    let small_baseline = params
        .flag("small_baseline_flag", false)
        .map_err(|error| failure(8, error))?;
    let dropped = if small_baseline {
        Vec::new()
    } else {
        params
            .indices("drop_ifg_index")
            .map_err(|error| failure(8, error))?
    };
    if dropped.iter().any(|&index| index >= loaded.n_ifg) {
        return Err(failure(8, "drop_ifg_index exceeds interferogram count"));
    }
    let unwrap = (0..loaded.n_ifg)
        .filter(|index| !dropped.contains(index))
        .collect::<Vec<_>>();
    let deramp = deramp_indices(&params, loaded.n_ifg).map_err(|error| failure(8, error))?;
    let result = estimate_scn(
        &ScnInputs {
            ph_uw: &loaded.ph_uw,
            xy: &loaded.xy,
            day: &loaded.day,
            n_ps: loaded.n_ps,
            n_ifg: loaded.n_ifg,
            ph_scla: loaded.ph_scla.as_deref(),
            c_ps_uw: loaded.c_ps_uw.as_deref(),
            scla_ramp: loaded.scla_ramp.as_deref(),
        },
        &ScnConfig {
            master_index: loaded.master_index,
            unwrap_indices: &unwrap,
            deramp_indices: &deramp,
            time_window: params
                .scalar("scn_time_win", 365.0)
                .map_err(|error| failure(8, error))?,
            wavelength: params
                .scalar("scn_wavelength", 100.0)
                .map_err(|error| failure(8, error))?,
        },
    )
    .map_err(|error| failure(8, error))?;
    write_output(root, loaded.n_ps, loaded.n_ifg, result).map_err(|error| failure(8, error))?;
    Ok(format!(
        "Stage 8 estimated spatially correlated noise for {} PS",
        loaded.n_ps
    ))
}

struct Loaded {
    n_ps: usize,
    n_ifg: usize,
    master_index: usize,
    ph_uw: Vec<f32>,
    xy: Vec<f64>,
    day: Vec<f64>,
    ph_scla: Option<Vec<f32>>,
    c_ps_uw: Option<Vec<f32>>,
    scla_ramp: Option<Vec<f64>>,
}

fn load(root: &Path) -> Result<Loaded, String> {
    let ps = read_required(root, "ps2.mat")?;
    let phase_file = read_required(root, "phuw2.mat")?;
    let dimensions = shape(&phase_file, "ph_uw")?;
    if dimensions.len() != 2 {
        return Err(format!("phuw2.ph_uw must be 2-D, found {dimensions:?}"));
    }
    let (n_ps, n_ifg) = (dimensions[0], dimensions[1]);
    let declared = integer_scalar(&ps, "n_ps")?;
    if declared != n_ps {
        return Err(format!(
            "ps2.n_ps={declared} does not match phuw2 rows={n_ps}"
        ));
    }
    let master_one = integer_scalar(&ps, "master_ix")?;
    if !(1..=n_ifg).contains(&master_one) {
        return Err("ps2.master_ix is outside phuw2 columns".to_owned());
    }
    let ph_uw = numeric_f32(&phase_file, "ph_uw")?;
    require_len("phuw2.ph_uw", ph_uw.len(), n_ps * n_ifg)?;
    let day = numeric_f64(&ps, "day")?;
    require_len("ps2.day", day.len(), n_ifg)?;
    let xy_full = numeric_f64(&ps, "xy")?;
    require_len("ps2.xy", xy_full.len(), n_ps * 3)?;
    let xy = xy_full
        .chunks_exact(3)
        .flat_map(|row| [row[1], row[2]])
        .collect::<Vec<_>>();
    let (ph_scla, c_ps_uw, scla_ramp) = load_scla(root, n_ps, n_ifg)?;
    Ok(Loaded {
        n_ps,
        n_ifg,
        master_index: master_one - 1,
        ph_uw,
        xy,
        day,
        ph_scla,
        c_ps_uw,
        scla_ramp,
    })
}

fn load_scla(
    root: &Path,
    n_ps: usize,
    n_ifg: usize,
) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f64>>), String> {
    if !root.join("scla2.mat").is_file() {
        return Ok((None, None, None));
    }
    let scla = read_required(root, "scla2.mat")?;
    let phase = numeric_f32(&scla, "ph_scla")?;
    require_len("scla2.ph_scla", phase.len(), n_ps * n_ifg)?;
    let constant = numeric_f32(&scla, "C_ps_uw")?;
    require_len("scla2.C_ps_uw", constant.len(), n_ps)?;
    let ramp = if scla.contains_key("ph_ramp") {
        let values = numeric_f64(&scla, "ph_ramp")?;
        if values.is_empty() {
            None
        } else {
            require_len("scla2.ph_ramp", values.len(), n_ps * n_ifg)?;
            Some(values)
        }
    } else {
        None
    };
    Ok((Some(phase), Some(constant), ramp))
}

fn deramp_indices(params: &Params, n_ifg: usize) -> Result<Vec<usize>, String> {
    match params.text("scn_deramp_ifg", "") {
        Ok(value) if value.eq_ignore_ascii_case("all") => Ok((0..n_ifg).collect()),
        Ok(value) if value.is_empty() => Ok(Vec::new()),
        Ok(value) => Err(format!("invalid scn_deramp_ifg text value {value}")),
        Err(_) => {
            let indices = params.indices("scn_deramp_ifg")?;
            if indices.iter().any(|&index| index >= n_ifg) {
                Err("scn_deramp_ifg exceeds interferogram count".to_owned())
            } else {
                Ok(indices)
            }
        }
    }
}

fn write_output(
    root: &Path,
    n_ps: usize,
    n_ifg: usize,
    output: pystamps_core::stage8::ScnOutputs,
) -> Result<(), String> {
    let transaction = StageTransaction::begin(root, "stage8").map_err(|error| error.to_string())?;
    let mut payload = BTreeMap::new();
    payload.insert(
        "ph_scn_slave".to_owned(),
        f64_array(vec![n_ps, n_ifg], output.ph_scn_slave),
    );
    payload.insert(
        "ph_hpt".to_owned(),
        f32_array(vec![n_ps, output.n_unwrap], output.ph_hpt),
    );
    payload.insert(
        "ph_ramp".to_owned(),
        f64_array(vec![n_ps, output.n_deramp], output.ph_ramp),
    );
    write_mat(transaction.path("scn2.mat"), &payload).map_err(|error| error.to_string())?;
    transaction
        .commit(&["scn2.mat"], "scn2.mat")
        .map_err(|error| error.to_string())
}

fn read_required(root: &Path, name: &str) -> Result<MatFile, String> {
    let path = root.join(name);
    if !path.is_file() {
        return Err(format!("missing required Stage 8 artifact {name}"));
    }
    read_mat(path).map_err(|error| error.to_string())
}

fn integer_scalar(file: &MatFile, key: &str) -> Result<usize, String> {
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

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), String> {
    (actual == expected)
        .then_some(())
        .ok_or_else(|| format!("{name} has {actual} values; expected {expected}"))
}
