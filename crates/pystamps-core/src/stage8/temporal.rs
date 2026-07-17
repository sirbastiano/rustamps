pub(super) fn gaussian_weights(
    day: &[f64],
    master_index: Option<usize>,
    time_window: f64,
) -> Result<Vec<f64>, String> {
    let width = day.len();
    let mut weights = vec![0.0; width * width];
    for target in 0..width {
        let row = &mut weights[target * width..(target + 1) * width];
        let mut maximum = f64::NEG_INFINITY;
        for source in 0..width {
            if Some(source) == master_index {
                row[source] = f64::NEG_INFINITY;
            } else {
                let scaled = (day[target] - day[source]) / time_window;
                row[source] = -0.5 * scaled * scaled;
                maximum = maximum.max(row[source]);
            }
        }
        if !maximum.is_finite() {
            return Err("temporal filtering needs a non-master interferogram".into());
        }
        let mut sum = 0.0;
        for value in row.iter_mut() {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        for value in row.iter_mut() {
            *value /= sum;
        }
    }
    Ok(weights)
}

pub(super) fn high_pass_into(values: &[f64], weights: &[f64], output: &mut [f64]) {
    let width = values.len();
    for target in 0..width {
        let low_pass: f64 = values
            .iter()
            .zip(&weights[target * width..(target + 1) * width])
            .map(|(value, weight)| value * weight)
            .sum();
        output[target] = values[target] - low_pass;
    }
}
