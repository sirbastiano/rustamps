use super::types::{CandidateStats, PrepError, SnapPrepInput};

fn check_len(field: &'static str, actual: usize, expected: usize) -> Result<(), PrepError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PrepError::LengthMismatch {
            field,
            expected,
            actual,
        })
    }
}

pub fn candidate_statistics(
    input: &SnapPrepInput<'_>,
    threshold: f64,
) -> Result<CandidateStats, PrepError> {
    let cells = input.shape.cells()?;
    if input.rslc_amplitudes.is_empty() {
        return Err(PrepError::EmptyAcquisitionStack);
    }
    check_len("longitude", input.lon.len(), cells)?;
    check_len("latitude", input.lat.len(), cells)?;
    check_len("height", input.height.len(), cells)?;
    for acquisition in input.rslc_amplitudes {
        check_len("RSLC amplitude", acquisition.len(), cells)?;
    }
    for interferogram in input.diff_phase {
        check_len("interferogram phase", interferogram.len(), cells)?;
    }

    let mut sum = vec![0.0_f64; cells];
    let mut sum_sq = vec![0.0_f64; cells];
    let mut has_low = vec![false; cells];
    for acquisition in input.rslc_amplitudes {
        let mut calibration_sum = 0.0;
        let mut calibration_count = 0usize;
        for &amplitude in *acquisition {
            if amplitude > 0.001 {
                calibration_sum += amplitude;
                calibration_count += 1;
            }
        }
        let calibration = if calibration_count == 0 {
            0.0
        } else {
            calibration_sum / calibration_count as f64
        };
        for cell in 0..cells {
            let normalized = acquisition[cell] / calibration;
            if normalized <= 0.00005 {
                has_low[cell] = true;
                sum[cell] = 0.0;
            } else {
                sum[cell] += normalized;
                sum_sq[cell] += normalized * normalized;
            }
        }
    }

    let count = input.rslc_amplitudes.len() as f64;
    let mut selected = vec![false; cells];
    let mut dispersion = vec![0.0_f32; cells];
    let mut normalized_sum = vec![0.0_f32; cells];
    for cell in 0..cells {
        let value = (count * sum_sq[cell] / (sum[cell] * sum[cell]) - 1.0)
            .max(0.0)
            .sqrt();
        dispersion[cell] = value as f32;
        normalized_sum[cell] = sum[cell] as f32;
        selected[cell] = input.lon[cell].is_finite()
            && input.lat[cell].is_finite()
            && input.height[cell].is_finite()
            && !has_low[cell]
            && value.is_finite()
            && sum[cell] > 0.0
            && value < threshold;
    }
    Ok(CandidateStats {
        selected,
        amplitude_dispersion: dispersion,
        normalized_amplitude_sum: normalized_sum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prep::types::{Complex32, RasterShape};

    #[test]
    fn rejects_a_pixel_with_one_near_zero_acquisition() {
        let acquisitions: Vec<Vec<f64>> = (0..10)
            .map(|index| vec![10.0, if index == 0 { 0.0 } else { 10.0 }])
            .collect();
        let refs: Vec<&[f64]> = acquisitions.iter().map(Vec::as_slice).collect();
        let phase = vec![Complex32::new(1.0, 0.0); 2];
        let phase_refs = vec![phase.as_slice()];
        let ones = vec![1.0_f32; 2];
        let input = SnapPrepInput {
            shape: RasterShape {
                length: 1,
                width: 2,
            },
            rslc_amplitudes: &refs,
            diff_phase: &phase_refs,
            lon: &ones,
            lat: &ones,
            height: &ones,
        };
        let stats = candidate_statistics(&input, 0.4).unwrap();
        assert_eq!(stats.selected, vec![true, false]);
        assert!((stats.amplitude_dispersion[1] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(stats.normalized_amplitude_sum, vec![10.0, 9.0]);
    }
}
