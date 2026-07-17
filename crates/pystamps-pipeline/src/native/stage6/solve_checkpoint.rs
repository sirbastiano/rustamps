use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pystamps_io::{read_mat, write_mat, MatFile, MatValue};

use super::super::mat::{f32_array, f64_array, scalar};
use super::grid::Grid;
use super::input::Input;

const SCHEMA_VERSION: u64 = 4;
const HASH_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0000_0100_0000_01b3;
const MATLAB_INTEGER_MASK: u64 = (1_u64 << 52) - 1;

pub struct Solution {
    pub values: Vec<f32>,
    pub msd: f64,
}

pub fn load(root: &Path, input: &Input, grid: &Grid, ordinal: usize) -> Option<Solution> {
    let path = path(root, input, ordinal);
    if !path.is_file() {
        return None;
    }
    let result = read_mat(&path)
        .map_err(|error| error.to_string())
        .and_then(|file| decode(&file, input, grid, ordinal));
    match result {
        Ok(solution) => Some(solution),
        Err(error) => {
            eprintln!(
                "Stage 6: ignoring invalid solve checkpoint {}: {error}",
                path.display()
            );
            None
        }
    }
}

pub fn write(
    root: &Path,
    input: &Input,
    grid: &Grid,
    ordinal: usize,
    solution: &Solution,
) -> Result<(), String> {
    validate_solution(solution, grid.n_points)?;
    let original = original_ifg(input, ordinal)?;
    let checksum = checksum(input, grid, ordinal, solution);
    let mut file = BTreeMap::new();
    for (name, value) in [
        ("pystamps_stage6_solve_schema", SCHEMA_VERSION),
        ("pystamps_input_fingerprint", input.fingerprint),
        ("solve_ordinal", ordinal as u64 + 1),
        ("ifg_index", original as u64 + 1),
        ("grid_rows", grid.rows as u64),
        ("grid_cols", grid.cols as u64),
        ("grid_n_points", grid.n_points as u64),
        ("payload_checksum", checksum),
    ] {
        file.insert(name.to_owned(), scalar(value as f64));
    }
    file.insert(
        "ph_uw".to_owned(),
        f32_array(vec![grid.n_points, 1], solution.values.clone()),
    );
    file.insert("msd".to_owned(), f64_array(vec![1, 1], vec![solution.msd]));
    write_mat(path(root, input, ordinal), &file).map_err(|error| error.to_string())
}

pub(super) fn path(root: &Path, input: &Input, ordinal: usize) -> PathBuf {
    let original = input.unwrap.get(ordinal).copied().unwrap_or(usize::MAX);
    root.join(".pystamps-stage6")
        .join(format!(
            "solve-v{SCHEMA_VERSION}-{:013x}",
            input.fingerprint
        ))
        .join(format!(
            "solve-{:06}-ifg-{:06}.mat",
            ordinal + 1,
            original.saturating_add(1)
        ))
}

fn decode(file: &MatFile, input: &Input, grid: &Grid, ordinal: usize) -> Result<Solution, String> {
    let original = original_ifg(input, ordinal)?;
    for (name, expected) in [
        ("pystamps_stage6_solve_schema", SCHEMA_VERSION),
        ("pystamps_input_fingerprint", input.fingerprint),
        ("solve_ordinal", ordinal as u64 + 1),
        ("ifg_index", original as u64 + 1),
        ("grid_rows", grid.rows as u64),
        ("grid_cols", grid.cols as u64),
        ("grid_n_points", grid.n_points as u64),
    ] {
        let found = integer(file, name)?;
        if found != expected {
            return Err(format!("{name}={found}, expected {expected}"));
        }
    }
    let values = match file.get("ph_uw") {
        Some(MatValue::F32(array)) if array.shape == [grid.n_points, 1] => array.values.clone(),
        Some(value) => {
            return Err(format!(
                "ph_uw has invalid type or shape {:?}",
                value.shape()
            ))
        }
        None => return Err("missing ph_uw".to_owned()),
    };
    let msd = match file.get("msd") {
        Some(MatValue::F64(array)) if array.shape == [1, 1] => array.values[0],
        Some(value) => return Err(format!("msd has invalid type or shape {:?}", value.shape())),
        None => return Err("missing msd".to_owned()),
    };
    let solution = Solution { values, msd };
    validate_solution(&solution, grid.n_points)?;
    let expected = checksum(input, grid, ordinal, &solution);
    let found = integer(file, "payload_checksum")?;
    if found != expected {
        return Err(format!(
            "payload checksum {found} does not match {expected}"
        ));
    }
    Ok(solution)
}

fn original_ifg(input: &Input, ordinal: usize) -> Result<usize, String> {
    input
        .unwrap
        .get(ordinal)
        .copied()
        .ok_or_else(|| format!("solve ordinal {ordinal} is outside the unwrap set"))
}

fn validate_solution(solution: &Solution, n_points: usize) -> Result<(), String> {
    if solution.values.len() != n_points {
        return Err(format!(
            "solve has {} grid values, expected {n_points}",
            solution.values.len()
        ));
    }
    if !solution.msd.is_finite() || solution.values.iter().any(|value| !value.is_finite()) {
        return Err("solve checkpoint contains non-finite values".to_owned());
    }
    Ok(())
}

fn integer(file: &MatFile, key: &str) -> Result<u64, String> {
    let value = match file.get(key) {
        Some(MatValue::F64(array)) if array.shape == [1, 1] => array.values[0],
        _ => return Err(format!("{key} is not an f64 scalar")),
    };
    if !value.is_finite()
        || value < 0.0
        || value > MATLAB_INTEGER_MASK as f64
        || value.fract() != 0.0
    {
        return Err(format!("{key} is not an exact non-negative MAT integer"));
    }
    Ok(value as u64)
}

fn checksum(input: &Input, grid: &Grid, ordinal: usize, solution: &Solution) -> u64 {
    let mut hash = HASH_OFFSET;
    for value in [
        SCHEMA_VERSION,
        input.fingerprint,
        ordinal as u64 + 1,
        input.unwrap[ordinal] as u64 + 1,
        grid.rows as u64,
        grid.cols as u64,
        grid.n_points as u64,
    ] {
        hash_u64(&mut hash, value);
    }
    for value in &solution.values {
        hash_u64(&mut hash, u64::from(value.to_bits()));
    }
    hash_u64(&mut hash, solution.msd.to_bits());
    (hash & MATLAB_INTEGER_MASK).max(1)
}

fn hash_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(HASH_PRIME);
    }
}
