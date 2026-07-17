use super::chronology::build_chronology;
use super::geometry::{local_xy, quantize_millimeters};
use super::types::{check_len, Matrix, Stage1Error, Stage1Input, Stage1Output};

fn select_rows<T: Copy>(matrix: &Matrix<T>, rows: &[usize]) -> Matrix<T> {
    let mut values = Vec::with_capacity(rows.len() * matrix.cols);
    for &row in rows {
        values.extend_from_slice(matrix.row(row));
    }
    Matrix {
        rows: rows.len(),
        cols: matrix.cols,
        values,
    }
}

fn select_vector<T: Copy>(values: &[T], rows: &[usize]) -> Vec<T> {
    rows.iter().map(|&row| values[row]).collect()
}

pub fn run_stage1(input: Stage1Input) -> Result<Stage1Output, Stage1Error> {
    if input.ij.cols != 3 {
        return Err(Stage1Error::InvalidMatrix("ij must have three columns"));
    }
    if input.lonlat.cols != 2 {
        return Err(Stage1Error::InvalidMatrix("lonlat must have two columns"));
    }
    let n_ps = input.ij.rows;
    check_len("phase rows", input.phase.rows, n_ps)?;
    check_len("lonlat rows", input.lonlat.rows, n_ps)?;
    if let Some(values) = &input.amplitude_dispersion {
        check_len("amplitude dispersion", values.len(), n_ps)?;
    }
    if let Some(values) = &input.height {
        check_len("height", values.len(), n_ps)?;
    }
    let chronology = build_chronology(
        &input.day_yyyymmdd,
        input.master_day_yyyymmdd,
        &input.bperp,
        &input.phase,
        input.per_pixel_bperp.as_ref(),
    )?;

    let mut valid_rows = Vec::new();
    for row in 0..n_ps {
        let valid_lonlat = input.lonlat.row(row).iter().all(|value| value.is_finite());
        let valid_phase = chronology
            .phase
            .row(row)
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite());
        if valid_lonlat && valid_phase {
            valid_rows.push(row);
        }
    }
    if valid_rows.is_empty() {
        return Err(Stage1Error::NoCandidates);
    }
    let valid_lonlat = valid_rows
        .iter()
        .map(|&row| [input.lonlat.row(row)[0], input.lonlat.row(row)[1]])
        .collect::<Vec<_>>();
    let (xy_valid, ll0) = local_xy(&valid_lonlat, input.heading_deg)?;
    let mut valid_order = (0..valid_rows.len()).collect::<Vec<_>>();
    valid_order.sort_by(|&left, &right| {
        (xy_valid[left][1] as f32)
            .total_cmp(&(xy_valid[right][1] as f32))
            .then_with(|| (xy_valid[left][0] as f32).total_cmp(&(xy_valid[right][0] as f32)))
    });
    let source_rows = valid_order
        .iter()
        .map(|&valid_index| valid_rows[valid_index])
        .collect::<Vec<_>>();
    let mut ij = select_rows(&input.ij, &source_rows);
    let lonlat = select_rows(&input.lonlat, &source_rows);
    for row in 0..source_rows.len() {
        ij.values[row * 3] = (row + 1) as f64;
    }
    let mut xy_values = Vec::with_capacity(source_rows.len() * 3);
    for (output_row, &valid_index) in valid_order.iter().enumerate() {
        xy_values.push((output_row + 1) as f32);
        xy_values.push(quantize_millimeters(xy_valid[valid_index][0]));
        xy_values.push(quantize_millimeters(xy_valid[valid_index][1]));
    }
    let phase = select_rows(&chronology.phase, &source_rows);
    let bperp_mat = if let Some(matrix) = &chronology.bperp_mat {
        select_rows(matrix, &source_rows)
    } else {
        let no_master = chronology
            .bperp
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (index + 1 != chronology.master_ix).then_some(value))
            .collect::<Vec<_>>();
        let mut values = Vec::with_capacity(source_rows.len() * no_master.len());
        for _ in &source_rows {
            values.extend_from_slice(&no_master);
        }
        Matrix {
            rows: source_rows.len(),
            cols: no_master.len(),
            values,
        }
    };
    Ok(Stage1Output {
        ij,
        phase,
        lonlat,
        xy: Matrix {
            rows: source_rows.len(),
            cols: 3,
            values: xy_values,
        },
        day: chronology.day,
        master_day: chronology.master_day,
        master_ix: chronology.master_ix,
        bperp: chronology.bperp,
        bperp_mat,
        sort_ix: source_rows.iter().map(|row| row + 1).collect(),
        ll0,
        amplitude_dispersion: input
            .amplitude_dispersion
            .as_ref()
            .map(|values| select_vector(values, &source_rows)),
        height: input
            .height
            .as_ref()
            .map(|values| select_vector(values, &source_rows)),
    })
}

#[cfg(test)]
mod tests {
    use super::super::types::Complex32;
    use super::*;

    #[test]
    fn filters_non_finite_rows_after_overwriting_null_master() {
        let nan = f32::NAN;
        let input = Stage1Input {
            ij: Matrix::new(
                3,
                3,
                vec![1.0, 10.0, 20.0, 2.0, 30.0, 40.0, 3.0, 50.0, 60.0],
            )
            .unwrap(),
            phase: Matrix::new(
                3,
                3,
                vec![
                    Complex32::new(2.0, 0.0),
                    Complex32::new(9.0, 0.0),
                    Complex32::new(3.0, 0.0),
                    Complex32::new(2.0, 0.0),
                    Complex32::new(9.0, 0.0),
                    Complex32::new(nan, 0.0),
                    Complex32::new(4.0, 0.0),
                    Complex32::new(nan, 0.0),
                    Complex32::new(6.0, 0.0),
                ],
            )
            .unwrap(),
            lonlat: Matrix::new(3, 2, vec![12.0, 45.0, 13.0, f64::INFINITY, 15.0, 48.0]).unwrap(),
            day_yyyymmdd: vec![20200101, 20200113, 20200125],
            master_day_yyyymmdd: 20200113,
            bperp: vec![10.0, 99.0, 30.0],
            per_pixel_bperp: None,
            amplitude_dispersion: Some(vec![1.0, 2.0, 3.0]),
            height: Some(vec![10.0, 20.0, 30.0]),
            heading_deg: None,
        };
        let output = run_stage1(input).unwrap();
        assert_eq!(output.phase.rows, 2);
        assert_eq!(output.sort_ix, vec![1, 3]);
        assert_eq!(output.phase.row(1)[1], Complex32::new(1.0, 0.0));
        assert_eq!(output.bperp_mat.values, vec![10.0, 30.0, 10.0, 30.0]);
        assert_eq!(output.height, Some(vec![10.0, 30.0]));
    }
}
