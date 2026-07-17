use super::Stage4Error;
use crate::stages::stage1::{Complex32, Matrix};
use delaunator::{triangulate, Point};
use std::collections::BTreeSet;

pub fn duplicate_keep(xy: &Matrix<f64>, coherence: &[f64]) -> Result<Vec<bool>, Stage4Error> {
    if xy.cols != 2 || xy.rows != coherence.len() {
        return Err(Stage4Error::InvalidInput("duplicate input shape mismatch"));
    }
    let mut keep = vec![true; xy.rows];
    let mut order = (0..xy.rows).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        xy.row(left)[0]
            .total_cmp(&xy.row(right)[0])
            .then_with(|| xy.row(left)[1].total_cmp(&xy.row(right)[1]))
            .then_with(|| left.cmp(&right))
    });
    let mut start = 0;
    while start < order.len() {
        let first = order[start];
        let mut end = start + 1;
        while end < order.len() && xy.row(order[end]) == xy.row(first) {
            end += 1;
        }
        let mut best = first;
        for &candidate in &order[start + 1..end] {
            if coherence[candidate] > coherence[best] {
                best = candidate;
            }
        }
        if end - start > 1 {
            for &duplicate in &order[start..end] {
                keep[duplicate] = duplicate == best;
            }
        }
        start = end;
    }
    Ok(keep)
}

pub fn adjacent_component_keep(
    ij_columns: &Matrix<i64>,
    coherence: &[f64],
) -> Result<Vec<bool>, Stage4Error> {
    if ij_columns.cols != 2 || ij_columns.rows != coherence.len() {
        return Err(Stage4Error::InvalidInput("adjacency input shape mismatch"));
    }
    if ij_columns.rows == 0 {
        return Ok(Vec::new());
    }
    let minimum_row = (0..ij_columns.rows)
        .map(|row| ij_columns.row(row)[0])
        .min()
        .unwrap();
    let minimum_col = (0..ij_columns.rows)
        .map(|row| ij_columns.row(row)[1])
        .min()
        .unwrap();
    let shifted = (0..ij_columns.rows)
        .map(|row| {
            [
                (ij_columns.row(row)[0] + 2 - minimum_row) as usize,
                (ij_columns.row(row)[1] + 2 - minimum_col) as usize,
            ]
        })
        .collect::<Vec<_>>();
    let rows = shifted.iter().map(|value| value[0]).max().unwrap() + 2;
    let cols = shifted.iter().map(|value| value[1]).max().unwrap() + 2;
    let mut neighborhood_owner = vec![0usize; rows * cols];
    for (index, &[row, col]) in shifted.iter().enumerate() {
        for adjacent_row in row - 1..=row + 1 {
            for adjacent_col in col - 1..=col + 1 {
                if adjacent_row == row && adjacent_col == col {
                    continue;
                }
                let cell = adjacent_row * cols + adjacent_col;
                if neighborhood_owner[cell] == 0 {
                    neighborhood_owner[cell] = index + 1;
                }
            }
        }
    }
    let mut linked = vec![Vec::new(); ij_columns.rows + 1];
    for (index, &[row, col]) in shifted.iter().enumerate() {
        let owner = neighborhood_owner[row * cols + col];
        if owner != 0 {
            linked[owner].push(index + 1);
        }
    }
    let mut keep = vec![true; ij_columns.rows];
    for start in 1..=ij_columns.rows {
        if linked[start].is_empty() {
            continue;
        }
        let mut component = vec![start];
        let mut cursor = 0;
        while cursor < component.len() {
            let node = component[cursor];
            component.extend(std::mem::take(&mut linked[node]));
            cursor += 1;
        }
        component.sort_unstable();
        component.dedup();
        let mut best = component[0];
        for &candidate in component.iter().skip(1) {
            if coherence[candidate - 1] > coherence[best - 1] {
                best = candidate;
            }
        }
        for node in component {
            if node != best {
                keep[node - 1] = false;
            }
        }
    }
    Ok(keep)
}

pub fn delaunay_edges(xy: &Matrix<f64>) -> Result<Vec<[usize; 2]>, Stage4Error> {
    if xy.cols != 2 {
        return Err(Stage4Error::InvalidInput(
            "triangulation xy must have two columns",
        ));
    }
    if xy.rows < 2 {
        return Ok(Vec::new());
    }
    if xy.rows == 2 {
        return Ok(vec![[0, 1]]);
    }
    let points = (0..xy.rows)
        .map(|row| Point {
            x: xy.row(row)[0],
            y: xy.row(row)[1],
        })
        .collect::<Vec<_>>();
    let triangulation = triangulate(&points);
    let mut edges = BTreeSet::new();
    for triangle in triangulation.triangles.chunks_exact(3) {
        for [a, b] in [
            [triangle[0], triangle[1]],
            [triangle[1], triangle[2]],
            [triangle[2], triangle[0]],
        ] {
            edges.insert(if a < b { [a, b] } else { [b, a] });
        }
    }
    if edges.is_empty() {
        for row in 0..xy.rows {
            let nearest = (0..xy.rows)
                .filter(|&candidate| candidate != row)
                .min_by(|&left, &right| {
                    let distance = |candidate: usize| {
                        (xy.row(row)[0] - xy.row(candidate)[0]).powi(2)
                            + (xy.row(row)[1] - xy.row(candidate)[1]).powi(2)
                    };
                    distance(left).total_cmp(&distance(right))
                })
                .unwrap();
            edges.insert(if row < nearest {
                [row, nearest]
            } else {
                [nearest, row]
            });
        }
    }
    Ok(edges.into_iter().collect())
}

pub fn phase_correction(
    phase: &Matrix<Complex32>,
    retain: &[bool],
    k_ps: &[f64],
    c_ps: &[f64],
    bperp: &[f64],
    small_baseline: bool,
    master_ix: usize,
) -> Result<Matrix<Complex32>, Stage4Error> {
    if phase.rows != retain.len()
        || phase.rows != k_ps.len()
        || phase.rows != c_ps.len()
        || phase.cols != bperp.len()
        || (!small_baseline && !(1..=phase.cols).contains(&master_ix))
    {
        return Err(Stage4Error::InvalidInput("phase-correction shape mismatch"));
    }
    let mut values = Vec::with_capacity(retain.iter().filter(|&&keep| keep).count() * phase.cols);
    for row in 0..phase.rows {
        if !retain[row] {
            continue;
        }
        for (col, &baseline) in bperp.iter().enumerate() {
            let angle = -k_ps[row] * baseline;
            let mut value =
                phase.row(row)[col] * Complex32::new(angle.cos() as f32, angle.sin() as f32);
            let magnitude = value.norm();
            if magnitude != 0.0 {
                value /= magnitude;
            }
            values.push(value);
        }
        if !small_baseline {
            let index = values.len() - phase.cols + master_ix - 1;
            values[index] = Complex32::new(c_ps[row].cos() as f32, c_ps[row].sin() as f32);
        }
    }
    Ok(Matrix {
        rows: retain.iter().filter(|&&keep| keep).count(),
        cols: phase.cols,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_tie_keeps_first_and_higher_coherence_replaces_it() {
        let xy = Matrix::new(3, 2, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            duplicate_keep(&xy, &[0.5, 0.5, 0.6]).unwrap(),
            vec![false, false, true]
        );
    }

    #[test]
    fn duplicate_filter_scales_without_changing_first_tie_semantics() {
        let rows = 20_000;
        let xy = Matrix::new(
            rows,
            2,
            (0..rows)
                .flat_map(|row| [(row / 2) as f64, (row / 2) as f64])
                .collect(),
        )
        .unwrap();
        let keep = duplicate_keep(&xy, &vec![0.5; rows]).unwrap();
        assert_eq!(keep.iter().filter(|&&value| value).count(), rows / 2);
        assert!(keep.chunks_exact(2).all(|pair| pair == [true, false]));
    }

    #[test]
    fn triangulation_emits_unique_undirected_edges() {
        let xy = Matrix::new(3, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(delaunay_edges(&xy).unwrap().len(), 3);
    }
}
