use std::collections::BTreeMap;
use std::path::Path;

use pystamps_io::{read_mat, write_mat_with_format, MatArray, MatFile, MatFormat, MatValue};

use super::super::mat::{
    complex32, complex32_array, f32_array, f64_array, numeric_f64, scalar, shape,
};
use super::cache_meta::{self, Checksum};
use super::grid::{select_phase, Grid};
use super::input::Input;

pub fn load(path: &Path, input: &Input) -> Result<Option<Grid>, String> {
    let file = read_mat(path).map_err(|error| error.to_string())?;
    if !cache_meta::matches(&file, input.fingerprint)? {
        return Ok(None);
    }
    let mask_shape = shape(&file, "nzix")?;
    if mask_shape.len() != 2 || mask_shape[0] == 0 || mask_shape[1] == 0 {
        return Err(format!("uw_grid.nzix has invalid shape {mask_shape:?}"));
    }
    let (rows, cols) = (mask_shape[0], mask_shape[1]);
    let mask = bool_values(&file, "nzix")?;
    let phase_shape = shape(&file, "ph")?;
    if phase_shape.len() != 2 || phase_shape[1] != input.unwrap.len() {
        return Err(format!("uw_grid.ph has invalid shape {phase_shape:?}"));
    }
    let n_points = phase_shape[0];
    if mask.iter().filter(|&&value| value).count() != n_points {
        return Err("uw_grid.nzix active count does not match uw_grid.ph rows".to_owned());
    }
    if shape(&file, "grid_ij")? != [input.n_ps, 2] {
        return Err("uw_grid.grid_ij has an incompatible shape".to_owned());
    }
    let coordinates = numeric_f64(&file, "grid_ij")?
        .chunks_exact(2)
        .map(|pair| one_based_pair(pair, rows, cols))
        .collect::<Result<Vec<_>, _>>()?;
    let phase_in = if file.contains_key("ph_in") {
        if shape(&file, "ph_in")? != [input.n_ps, input.unwrap.len()] {
            return Err("uw_grid.ph_in has an incompatible shape".to_owned());
        }
        complex32(&file, "ph_in")?
    } else {
        select_phase(input)
    };
    let grid = Grid {
        fingerprint: input.fingerprint,
        rows,
        cols,
        mask,
        coordinates,
        phase: complex32(&file, "ph")?,
        phase_in,
        n_points,
        min_x: optional_scalar(&file, "grid_x_min")?.unwrap_or(0.0) as f32,
        min_y: optional_scalar(&file, "grid_y_min")?.unwrap_or(0.0) as f32,
    };
    cache_meta::validate(&file, payload_checksum(&grid), "uw_grid")?;
    Ok(Some(grid))
}

pub fn write(path: &Path, input: &Input, grid: &Grid) -> Result<(), String> {
    let mut file = BTreeMap::new();
    cache_meta::insert(&mut file, input.fingerprint, payload_checksum(grid));
    file.insert(
        "ph".to_owned(),
        complex32_array(vec![grid.n_points, input.unwrap.len()], grid.phase.clone()),
    );
    file.insert(
        "ph_in".to_owned(),
        complex32_array(vec![input.n_ps, input.unwrap.len()], grid.phase_in.clone()),
    );
    for name in ["ph_lowpass", "ph_uw_predef", "ph_in_predef"] {
        file.insert(name.to_owned(), complex32_array(vec![0, 0], Vec::new()));
    }
    let active = (0..grid.cols)
        .flat_map(|col| {
            (0..grid.rows)
                .filter_map(move |row| grid.mask[row * grid.cols + col].then_some((row, col)))
        })
        .collect::<Vec<_>>();
    file.insert(
        "xy".to_owned(),
        f64_array(
            vec![grid.n_points, 3],
            active
                .iter()
                .enumerate()
                .flat_map(|(id, &(row, col))| {
                    [
                        (id + 1) as f64,
                        (col as f64 + 0.5) * input.options.grid_size,
                        (row as f64 + 0.5) * input.options.grid_size,
                    ]
                })
                .collect(),
        ),
    );
    file.insert(
        "ij".to_owned(),
        f64_array(
            vec![grid.n_points, 2],
            active
                .iter()
                .flat_map(|&(row, col)| [(row + 1) as f64, (col + 1) as f64])
                .collect(),
        ),
    );
    file.insert(
        "nzix".to_owned(),
        MatValue::Bool(MatArray {
            shape: vec![grid.rows, grid.cols],
            values: grid.mask.clone(),
        }),
    );
    file.insert(
        "grid_ij".to_owned(),
        f64_array(
            vec![input.n_ps, 2],
            grid.coordinates
                .iter()
                .flat_map(|point| [(point[0] + 1) as f64, (point[1] + 1) as f64])
                .collect(),
        ),
    );
    for (name, value) in [
        ("n_i", grid.rows as f64),
        ("n_j", grid.cols as f64),
        ("n_ifg", input.unwrap.len() as f64),
        ("n_ps", grid.n_points as f64),
        ("pix_size", input.options.grid_size),
    ] {
        file.insert(name.to_owned(), scalar(value));
    }
    file.insert(
        "grid_x_min".to_owned(),
        f32_array(vec![1, 1], vec![grid.min_x]),
    );
    file.insert(
        "grid_y_min".to_owned(),
        f32_array(vec![1, 1], vec![grid.min_y]),
    );
    write_mat_with_format(path, &file, MatFormat::V73).map_err(|error| error.to_string())
}

fn one_based_pair(values: &[f64], rows: usize, cols: usize) -> Result<[usize; 2], String> {
    if values.len() != 2
        || values[0].fract() != 0.0
        || values[1].fract() != 0.0
        || !(1.0..=rows as f64).contains(&values[0])
        || !(1.0..=cols as f64).contains(&values[1])
    {
        return Err("uw_grid.grid_ij contains an invalid one-based coordinate".to_owned());
    }
    Ok([values[0] as usize - 1, values[1] as usize - 1])
}

fn bool_values(file: &MatFile, key: &str) -> Result<Vec<bool>, String> {
    match file
        .get(key)
        .ok_or_else(|| format!("missing MAT key {key}"))?
    {
        MatValue::Bool(array) => Ok(array.values.clone()),
        MatValue::U8(array) => Ok(array.values.iter().map(|&value| value != 0).collect()),
        MatValue::I8(array) => Ok(array.values.iter().map(|&value| value != 0).collect()),
        _ => Err(format!("MAT key {key} is not logical")),
    }
}

fn optional_scalar(file: &MatFile, key: &str) -> Result<Option<f64>, String> {
    if !file.contains_key(key) {
        return Ok(None);
    }
    numeric_f64(file, key)?
        .first()
        .copied()
        .map(Some)
        .ok_or_else(|| format!("{key} is empty"))
}

fn payload_checksum(grid: &Grid) -> u64 {
    let mut checksum = Checksum::new(grid.fingerprint);
    for value in [grid.rows, grid.cols, grid.n_points] {
        checksum.usize(value);
    }
    checksum.usize(grid.mask.len());
    grid.mask.iter().for_each(|&value| checksum.bool(value));
    checksum.usize(grid.coordinates.len());
    for point in &grid.coordinates {
        checksum.usize(point[0]);
        checksum.usize(point[1]);
    }
    checksum.usize(grid.phase.len());
    grid.phase
        .iter()
        .for_each(|&value| checksum.complex32(value));
    checksum.usize(grid.phase_in.len());
    grid.phase_in
        .iter()
        .for_each(|&value| checksum.complex32(value));
    checksum.f32(grid.min_x);
    checksum.f32(grid.min_y);
    checksum.finish()
}
