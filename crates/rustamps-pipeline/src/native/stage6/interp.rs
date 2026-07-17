use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rustamps_io::{read_mat, write_mat};

use super::super::mat::{f64_array, numeric_f64, scalar, shape};
use super::cache_meta::{self, Checksum};
use super::grid::Grid;
use super::nearest::nearest_grid;

pub struct Interpolation {
    pub edges: Vec<[usize; 2]>,
    pub row_indices: Vec<f64>,
    pub col_indices: Vec<f64>,
    pub nearest: Vec<usize>,
}

pub fn load_or_build(root: &Path, grid: &Grid) -> Result<Interpolation, String> {
    let path = root.join("uw_interp.mat");
    if path.is_file() {
        if let Some(interpolation) = load(&path, grid)? {
            return Ok(interpolation);
        }
    }
    let interpolation = build(grid)?;
    write(&path, grid, &interpolation)?;
    Ok(interpolation)
}

fn load(path: &Path, grid: &Grid) -> Result<Option<Interpolation>, String> {
    let file = read_mat(path).map_err(|error| error.to_string())?;
    if !cache_meta::matches(&file, grid.fingerprint)? {
        return Ok(None);
    }
    let edge_shape = shape(&file, "edgs")?;
    if edge_shape.len() != 2 || edge_shape[1] != 3 {
        return Err(format!("uw_interp.edgs has invalid shape {edge_shape:?}"));
    }
    let edges = numeric_f64(&file, "edgs")?
        .chunks_exact(3)
        .enumerate()
        .map(|(index, row)| {
            if row[0] != (index + 1) as f64 {
                return Err("uw_interp.edgs identifiers are not sequential".to_owned());
            }
            Ok([
                one_based_index(row[1], grid.n_points, "uw_interp.edgs")?,
                one_based_index(row[2], grid.n_points, "uw_interp.edgs")?,
            ])
        })
        .collect::<Result<Vec<_>, String>>()?;
    let expected_row = [grid.rows.saturating_sub(1), grid.cols];
    let expected_col = [grid.rows, grid.cols.saturating_sub(1)];
    if shape(&file, "rowix")? != expected_row || shape(&file, "colix")? != expected_col {
        return Err("uw_interp rowix/colix shapes do not match uw_grid.nzix".to_owned());
    }
    let row_indices = numeric_f64(&file, "rowix")?;
    let col_indices = numeric_f64(&file, "colix")?;
    validate_edge_indices(&row_indices, edges.len())?;
    validate_edge_indices(&col_indices, edges.len())?;
    if shape(&file, "Z")? != [grid.rows, grid.cols] {
        return Err("uw_interp.Z shape does not match uw_grid.nzix".to_owned());
    }
    let nearest = numeric_f64(&file, "Z")?
        .into_iter()
        .map(|value| one_based_index(value, grid.n_points, "uw_interp.Z"))
        .collect::<Result<Vec<_>, _>>()?;
    let interpolation = Interpolation {
        edges,
        row_indices,
        col_indices,
        nearest,
    };
    cache_meta::validate(
        &file,
        payload_checksum(grid.fingerprint, &interpolation),
        "uw_interp",
    )?;
    Ok(Some(interpolation))
}

fn build(grid: &Grid) -> Result<Interpolation, String> {
    let mut points = Vec::with_capacity(grid.n_points);
    for col in 0..grid.cols {
        for row in 0..grid.rows {
            if grid.mask[row * grid.cols + col] {
                points.push([col as f64, row as f64]);
            }
        }
    }
    let nearest = nearest_grid(&points, grid.rows, grid.cols)?;
    let mut pairs = BTreeSet::new();
    for row in 0..grid.rows {
        for col in 0..grid.cols.saturating_sub(1) {
            insert_pair(
                &mut pairs,
                nearest[row * grid.cols + col],
                nearest[row * grid.cols + col + 1],
            );
        }
    }
    for row in 0..grid.rows.saturating_sub(1) {
        for col in 0..grid.cols {
            insert_pair(
                &mut pairs,
                nearest[row * grid.cols + col],
                nearest[(row + 1) * grid.cols + col],
            );
        }
    }
    let edges = pairs.into_iter().map(|(a, b)| [a, b]).collect::<Vec<_>>();
    let ids = edges
        .iter()
        .enumerate()
        .map(|(index, edge)| ((edge[0], edge[1]), index + 1))
        .collect::<BTreeMap<_, _>>();
    let mut row_indices = Vec::with_capacity(grid.rows.saturating_sub(1) * grid.cols);
    for row in 0..grid.rows.saturating_sub(1) {
        for col in 0..grid.cols {
            row_indices.push(signed_id(
                nearest[row * grid.cols + col],
                nearest[(row + 1) * grid.cols + col],
                &ids,
            ));
        }
    }
    let mut col_indices = Vec::with_capacity(grid.rows * grid.cols.saturating_sub(1));
    for row in 0..grid.rows {
        for col in 0..grid.cols.saturating_sub(1) {
            col_indices.push(signed_id(
                nearest[row * grid.cols + col],
                nearest[row * grid.cols + col + 1],
                &ids,
            ));
        }
    }
    Ok(Interpolation {
        edges,
        row_indices,
        col_indices,
        nearest,
    })
}

fn write(path: &Path, grid: &Grid, interpolation: &Interpolation) -> Result<(), String> {
    let mut file = BTreeMap::new();
    cache_meta::insert(
        &mut file,
        grid.fingerprint,
        payload_checksum(grid.fingerprint, interpolation),
    );
    file.insert(
        "edgs".to_owned(),
        f64_array(
            vec![interpolation.edges.len(), 3],
            interpolation
                .edges
                .iter()
                .enumerate()
                .flat_map(|(index, edge)| {
                    [
                        (index + 1) as f64,
                        (edge[0] + 1) as f64,
                        (edge[1] + 1) as f64,
                    ]
                })
                .collect(),
        ),
    );
    file.insert(
        "n_edge".to_owned(),
        scalar(interpolation.edges.len() as f64),
    );
    file.insert(
        "rowix".to_owned(),
        f64_array(
            vec![grid.rows.saturating_sub(1), grid.cols],
            interpolation.row_indices.clone(),
        ),
    );
    file.insert(
        "colix".to_owned(),
        f64_array(
            vec![grid.rows, grid.cols.saturating_sub(1)],
            interpolation.col_indices.clone(),
        ),
    );
    file.insert(
        "Z".to_owned(),
        f64_array(
            vec![grid.rows, grid.cols],
            interpolation
                .nearest
                .iter()
                .map(|&index| (index + 1) as f64)
                .collect(),
        ),
    );
    write_mat(path, &file).map_err(|error| error.to_string())
}

fn insert_pair(edges: &mut BTreeSet<(usize, usize)>, a: usize, b: usize) {
    if a != b {
        edges.insert((a.min(b), a.max(b)));
    }
}

fn signed_id(a: usize, b: usize, ids: &BTreeMap<(usize, usize), usize>) -> f64 {
    if a == b {
        0.0
    } else {
        let id = ids[&(a.min(b), a.max(b))] as f64;
        if a < b {
            id
        } else {
            -id
        }
    }
}

fn one_based_index(value: f64, count: usize, name: &str) -> Result<usize, String> {
    if !value.is_finite() || value.fract() != 0.0 || !(1.0..=count as f64).contains(&value) {
        Err(format!(
            "{name} contains an invalid one-based index {value}"
        ))
    } else {
        Ok(value as usize - 1)
    }
}

fn validate_edge_indices(values: &[f64], count: usize) -> Result<(), String> {
    if values.iter().any(|value| {
        !value.is_nan()
            && (!value.is_finite()
                || (*value != 0.0 && (value.fract() != 0.0 || value.abs() > count as f64)))
    }) {
        Err("uw_interp edge-index matrix references a missing edge".to_owned())
    } else {
        Ok(())
    }
}

fn payload_checksum(fingerprint: u64, interpolation: &Interpolation) -> u64 {
    let mut checksum = Checksum::new(fingerprint);
    checksum.usize(interpolation.edges.len());
    for edge in &interpolation.edges {
        checksum.usize(edge[0]);
        checksum.usize(edge[1]);
    }
    checksum.usize(interpolation.row_indices.len());
    for &value in &interpolation.row_indices {
        checksum.f64(value);
    }
    checksum.usize(interpolation.col_indices.len());
    for &value in &interpolation.col_indices {
        checksum.f64(value);
    }
    checksum.usize(interpolation.nearest.len());
    interpolation
        .nearest
        .iter()
        .for_each(|&value| checksum.usize(value));
    checksum.finish()
}
