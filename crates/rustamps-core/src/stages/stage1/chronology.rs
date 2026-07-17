use super::types::{check_len, Complex32, Matrix, Stage1Error};

#[derive(Clone, Debug, PartialEq)]
pub struct Chronology {
    pub day: Vec<f64>,
    pub master_day: f64,
    pub master_ix: usize,
    pub bperp: Vec<f32>,
    pub phase: Matrix<Complex32>,
    pub bperp_mat: Option<Matrix<f32>>,
}

fn leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

pub fn matlab_datenum(yyyymmdd: i32) -> Result<f64, Stage1Error> {
    let year = yyyymmdd / 10_000;
    let month = (yyyymmdd / 100) % 100;
    let day = yyyymmdd % 100;
    let month_days = [
        31,
        28 + i32::from(leap_year(year)),
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if year < 1 || !(1..=12).contains(&month) || day < 1 || day > month_days[(month - 1) as usize] {
        return Err(Stage1Error::InvalidDate(yyyymmdd));
    }
    let prior_year = year - 1;
    let days_before_year = 365 * prior_year + prior_year / 4 - prior_year / 100 + prior_year / 400;
    let days_before_month: i32 = month_days[..(month - 1) as usize].iter().sum();
    Ok((days_before_year + days_before_month + day + 366) as f64)
}

pub fn build_chronology(
    day_yyyymmdd: &[i32],
    master_yyyymmdd: i32,
    bperp: &[f64],
    phase: &Matrix<Complex32>,
    per_pixel_bperp: Option<&Matrix<f32>>,
) -> Result<Chronology, Stage1Error> {
    check_len("bperp", bperp.len(), day_yyyymmdd.len())?;
    check_len("phase columns", phase.cols, day_yyyymmdd.len())?;
    if let Some(matrix) = per_pixel_bperp {
        check_len("per-pixel bperp rows", matrix.rows, phase.rows)?;
        check_len("per-pixel bperp columns", matrix.cols, day_yyyymmdd.len())?;
    }
    let mut dated = day_yyyymmdd
        .iter()
        .enumerate()
        .map(|(index, &date)| Ok((matlab_datenum(date)?, index)))
        .collect::<Result<Vec<_>, Stage1Error>>()?;
    dated.sort_by(|left, right| left.0.total_cmp(&right.0));
    let master_day = matlab_datenum(master_yyyymmdd)?;
    let matches = dated.iter().filter(|(day, _)| *day == master_day).count();
    if matches > 1 {
        return Err(Stage1Error::DuplicateMaster);
    }
    let has_master = matches == 1;
    let master_zero = if has_master {
        dated
            .iter()
            .position(|(day, _)| *day == master_day)
            .unwrap()
    } else {
        dated.iter().filter(|(day, _)| *day < master_day).count()
    };
    let full_cols = dated.len() + usize::from(!has_master);
    let mut day = Vec::with_capacity(full_cols);
    let mut baseline = Vec::with_capacity(full_cols);
    for (sorted_col, &(date, source_col)) in dated.iter().enumerate() {
        if !has_master && sorted_col == master_zero {
            day.push(master_day);
            baseline.push(0.0);
        }
        day.push(date);
        baseline.push(if has_master && sorted_col == master_zero {
            0.0
        } else {
            bperp[source_col] as f32
        });
    }
    if !has_master && master_zero == dated.len() {
        day.push(master_day);
        baseline.push(0.0);
    }

    let mut phase_out = Vec::with_capacity(phase.rows * full_cols);
    for row in 0..phase.rows {
        for (sorted_col, &(_, source_col)) in dated.iter().enumerate() {
            if !has_master && sorted_col == master_zero {
                phase_out.push(Complex32::new(1.0, 0.0));
            }
            phase_out.push(if has_master && sorted_col == master_zero {
                Complex32::new(1.0, 0.0)
            } else {
                phase.row(row)[source_col]
            });
        }
        if !has_master && master_zero == dated.len() {
            phase_out.push(Complex32::new(1.0, 0.0));
        }
    }

    let bperp_mat = per_pixel_bperp.map(|matrix| {
        let mut values = Vec::with_capacity(matrix.rows * (full_cols - 1));
        for row in 0..matrix.rows {
            for (sorted_col, &(_, source_col)) in dated.iter().enumerate() {
                if has_master && sorted_col == master_zero {
                    continue;
                }
                values.push(matrix.row(row)[source_col]);
            }
        }
        Matrix {
            rows: matrix.rows,
            cols: full_cols - 1,
            values,
        }
    });
    Ok(Chronology {
        day,
        master_day,
        master_ix: master_zero + 1,
        bperp: baseline,
        phase: Matrix {
            rows: phase.rows,
            cols: full_cols,
            values: phase_out,
        },
        bperp_mat,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_null_master_is_overwritten_not_inserted() {
        let phase = Matrix::new(
            1,
            3,
            vec![
                Complex32::new(2.0, 0.0),
                Complex32::new(9.0, 0.0),
                Complex32::new(3.0, 0.0),
            ],
        )
        .unwrap();
        let result = build_chronology(
            &[20200101, 20200113, 20200125],
            20200113,
            &[10.0, 99.0, 30.0],
            &phase,
            None,
        )
        .unwrap();
        assert_eq!(result.master_ix, 2);
        assert_eq!(result.phase.cols, 3);
        assert_eq!(result.phase.values[1], Complex32::new(1.0, 0.0));
        assert_eq!(result.bperp, vec![10.0, 0.0, 30.0]);
        assert_eq!(matlab_datenum(19700101).unwrap(), 719_529.0);
    }
}
