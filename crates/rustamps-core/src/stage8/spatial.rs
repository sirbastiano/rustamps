use std::collections::HashMap;

use rayon::prelude::*;

pub(super) fn gaussian_low_pass(
    phase: &[f32],
    xy: &[f64],
    n_ps: usize,
    width: usize,
    wavelength: f64,
) -> Result<Vec<f64>, String> {
    if phase.len() != n_ps * width || xy.len() != n_ps * 2 || width == 0 {
        return Err("spatial filter matrix shape is invalid".into());
    }
    let radius = 4.0 * wavelength;
    let radius_sq = radius * radius;
    let gaussian_denominator = 2.0 * wavelength * wavelength;
    let mut bins: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let mut point_bins = Vec::with_capacity(n_ps);
    for point in 0..n_ps {
        let key = bin_key(xy[point * 2], xy[point * 2 + 1], radius)?;
        bins.entry(key).or_default().push(point);
        point_bins.push(key);
    }

    let mut output = vec![0.0; n_ps * width];
    output.par_chunks_mut(width).enumerate().for_each_init(
        || Vec::<usize>::with_capacity(128),
        |neighbors, (point, row_output)| {
            neighbors.clear();
            let (bin_x, bin_y) = point_bins[point];
            for offset_x in -1_i64..=1 {
                for offset_y in -1_i64..=1 {
                    if let Some(points) = bins.get(&(bin_x + offset_x, bin_y + offset_y)) {
                        neighbors.extend_from_slice(points);
                    }
                }
            }
            neighbors.sort_unstable();

            let x = xy[point * 2];
            let y = xy[point * 2 + 1];
            let mut weight_sum = 0.0;
            for &neighbor in neighbors.iter() {
                let dx = xy[neighbor * 2] - x;
                let dy = xy[neighbor * 2 + 1] - y;
                let distance_sq = dx * dx + dy * dy;
                if distance_sq >= radius_sq {
                    continue;
                }
                let weight = (-distance_sq / gaussian_denominator).exp();
                weight_sum += weight;
                let neighbor_phase = &phase[neighbor * width..(neighbor + 1) * width];
                for column in 0..width {
                    row_output[column] += weight * neighbor_phase[column] as f64;
                }
            }
            for value in row_output.iter_mut() {
                *value /= weight_sum;
            }
        },
    );

    let reference = output[..width].to_vec();
    output.par_chunks_mut(width).for_each(|row| {
        for column in 0..width {
            row[column] -= reference[column];
        }
    });
    Ok(output)
}

fn bin_key(x: f64, y: f64, width: f64) -> Result<(i64, i64), String> {
    let bin_x = (x / width).floor();
    let bin_y = (y / width).floor();
    let safe_min = i64::MIN as f64 + 2.0;
    let safe_max = i64::MAX as f64 - 2.0;
    if bin_x < safe_min || bin_x > safe_max || bin_y < safe_min || bin_y > safe_max {
        return Err("spatial coordinate exceeds hash-grid range".into());
    }
    Ok((bin_x as i64, bin_y as i64))
}
