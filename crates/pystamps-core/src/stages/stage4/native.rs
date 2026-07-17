use super::{
    adjacent_component_keep, delaunay_edges, duplicate_keep, edge_noise_statistics,
    phase_correction, Stage4Error, Stage4Measurements,
};
use crate::stages::stage1::{Complex32, Matrix};

#[derive(Clone, Debug, PartialEq)]
pub struct NativeStage4Options {
    pub weed_neighbours: bool,
    pub weed_zero_elevation: bool,
    pub weed_standard_dev: f64,
    pub weed_max_noise: f64,
    pub weed_time_window: f64,
    pub small_baseline: bool,
    pub master_ix: usize,
    pub interferogram_indices: Vec<usize>,
}

impl Default for NativeStage4Options {
    fn default() -> Self {
        Self {
            weed_neighbours: false,
            weed_zero_elevation: false,
            weed_standard_dev: 1.0,
            weed_max_noise: f64::INFINITY,
            weed_time_window: 730.0,
            small_baseline: false,
            master_ix: 1,
            interferogram_indices: Vec::new(),
        }
    }
}

fn select_rows<T: Copy>(matrix: &Matrix<T>, keep: &[bool]) -> Matrix<T> {
    let mut values = Vec::with_capacity(keep.iter().filter(|&&value| value).count() * matrix.cols);
    for (row, &selected) in keep.iter().enumerate() {
        if selected {
            values.extend_from_slice(matrix.row(row));
        }
    }
    Matrix {
        rows: keep.iter().filter(|&&value| value).count(),
        cols: matrix.cols,
        values,
    }
}

fn select_columns<T: Copy>(matrix: &Matrix<T>, columns: &[usize]) -> Matrix<T> {
    let mut values = Vec::with_capacity(matrix.rows * columns.len());
    for row in 0..matrix.rows {
        for &column in columns {
            values.push(matrix.row(row)[column]);
        }
    }
    Matrix {
        rows: matrix.rows,
        cols: columns.len(),
        values,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn measure_stage4(
    ij_columns: &Matrix<i64>,
    xy: &Matrix<f64>,
    coherence: &[f64],
    height: Option<&[f32]>,
    phase: &Matrix<Complex32>,
    k_ps: &[f64],
    c_ps: &[f64],
    bperp: &[f64],
    day: &[f64],
    options: &NativeStage4Options,
) -> Result<Stage4Measurements, Stage4Error> {
    let rows = coherence.len();
    if ij_columns.rows != rows
        || ij_columns.cols != 2
        || xy.rows != rows
        || xy.cols != 2
        || phase.rows != rows
        || k_ps.len() != rows
        || c_ps.len() != rows
        || bperp.len() != phase.cols
        || (!options.small_baseline && day.len() != phase.cols)
        || height.is_some_and(|values| values.len() != rows)
        || options
            .interferogram_indices
            .iter()
            .any(|&index| index >= phase.cols)
    {
        return Err(Stage4Error::InvalidInput(
            "native measurement arrays are not row aligned",
        ));
    }
    let adjacency_keep = if options.weed_neighbours {
        adjacent_component_keep(ij_columns, coherence)?
    } else {
        vec![true; rows]
    };
    let height_keep = (0..rows)
        .map(|row| !options.weed_zero_elevation || height.is_some_and(|values| values[row] >= 1e-6))
        .collect::<Vec<_>>();
    if options.weed_zero_elevation && height.is_none() {
        return Err(Stage4Error::InvalidInput(
            "height is required for zero-elevation weeding",
        ));
    }
    let before_duplicates = adjacency_keep
        .iter()
        .zip(&height_keep)
        .map(|(&adjacent, &elevated)| adjacent && elevated)
        .collect::<Vec<_>>();
    let xy_active = select_rows(xy, &before_duplicates);
    let coherence_active = coherence
        .iter()
        .zip(&before_duplicates)
        .filter_map(|(&value, &keep)| keep.then_some(value))
        .collect::<Vec<_>>();
    let active_duplicates = duplicate_keep(&xy_active, &coherence_active)?;
    let mut duplicate_keep_all = vec![true; rows];
    let mut active_row = 0;
    for row in 0..rows {
        if before_duplicates[row] {
            duplicate_keep_all[row] = active_duplicates[active_row];
            active_row += 1;
        }
    }
    let spatial_keep = (0..rows)
        .map(|row| before_duplicates[row] && duplicate_keep_all[row])
        .collect::<Vec<_>>();
    let retained = spatial_keep.iter().filter(|&&keep| keep).count();
    if options.weed_standard_dev >= std::f64::consts::PI
        && options.weed_max_noise >= std::f64::consts::PI
    {
        return Ok(Stage4Measurements {
            adjacency_keep,
            duplicate_keep: duplicate_keep_all,
            ps_std: vec![0.0; retained],
            ps_max: vec![0.0; retained],
        });
    }
    let corrected = phase_correction(
        phase,
        &spatial_keep,
        k_ps,
        c_ps,
        bperp,
        options.small_baseline,
        options.master_ix,
    )?;
    let xy_retained = select_rows(xy, &spatial_keep);
    let edges = delaunay_edges(&xy_retained)?;
    let columns = if options.interferogram_indices.is_empty() {
        (0..phase.cols).collect::<Vec<_>>()
    } else {
        options.interferogram_indices.clone()
    };
    let corrected = select_columns(&corrected, &columns);
    let bperp = columns
        .iter()
        .map(|&column| bperp[column])
        .collect::<Vec<_>>();
    let day = if options.small_baseline {
        Vec::new()
    } else {
        columns.iter().map(|&column| day[column]).collect()
    };
    let noise = edge_noise_statistics(
        &corrected,
        &edges,
        &bperp,
        &day,
        options.weed_time_window,
        options.small_baseline,
    )?;
    Ok(Stage4Measurements {
        adjacency_keep,
        duplicate_keep: duplicate_keep_all,
        ps_std: noise.ps_std,
        ps_max: noise.ps_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_noise_weeding_skips_expensive_edge_statistics() {
        let result = measure_stage4(
            &Matrix::new(1, 2, vec![0, 0]).unwrap(),
            &Matrix::new(1, 2, vec![0.0, 0.0]).unwrap(),
            &[0.8],
            None,
            &Matrix::new(1, 1, vec![Complex32::new(1.0, 0.0)]).unwrap(),
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &NativeStage4Options {
                weed_standard_dev: std::f64::consts::PI,
                weed_max_noise: std::f64::consts::PI,
                ..NativeStage4Options::default()
            },
        )
        .unwrap();
        assert_eq!(result.ps_std, vec![0.0]);
    }
}
