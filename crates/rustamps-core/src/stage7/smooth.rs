use super::Stage7Error;

pub fn smooth_neighbor_envelope(
    k_ps_uw: &[f64],
    c_ps_uw: &[f32],
    edges: &[[usize; 2]],
) -> Result<(Vec<f64>, Vec<f32>), Stage7Error> {
    if k_ps_uw.len() != c_ps_uw.len() {
        return Err(Stage7Error::new(
            "K and C arrays must have matching lengths",
        ));
    }
    let n_ps = k_ps_uw.len();
    let mut k_min = vec![f64::INFINITY; n_ps];
    let mut k_max = vec![f64::NEG_INFINITY; n_ps];
    let mut c_min = vec![f32::INFINITY; n_ps];
    let mut c_max = vec![f32::NEG_INFINITY; n_ps];

    for &[left, right] in edges {
        if left >= n_ps || right >= n_ps || left == right {
            return Err(Stage7Error::new("SCLA smoothing edge is invalid"));
        }
        update_bounds(&mut k_min[left], &mut k_max[left], k_ps_uw[right]);
        update_bounds(&mut k_min[right], &mut k_max[right], k_ps_uw[left]);
        update_bounds(&mut c_min[left], &mut c_max[left], c_ps_uw[right]);
        update_bounds(&mut c_min[right], &mut c_max[right], c_ps_uw[left]);
    }

    let mut k_output = k_ps_uw.to_vec();
    let mut c_output = c_ps_uw.to_vec();
    for index in 0..n_ps {
        if k_max[index].is_finite() && k_output[index] > k_max[index] {
            k_output[index] = k_max[index];
        }
        if k_min[index].is_finite() && k_output[index] < k_min[index] {
            k_output[index] = k_min[index];
        }
        if c_max[index].is_finite() && c_output[index] > c_max[index] {
            c_output[index] = c_max[index];
        }
        if c_min[index].is_finite() && c_output[index] < c_min[index] {
            c_output[index] = c_min[index];
        }
    }
    Ok((k_output, c_output))
}

pub fn build_smoothed_phase(
    k_ps_uw: &[f64],
    bperp_mat: &[f64],
    n_ifg: usize,
) -> Result<Vec<f32>, Stage7Error> {
    if n_ifg == 0 || bperp_mat.len() != k_ps_uw.len() * n_ifg {
        return Err(Stage7Error::new("smoothed phase matrix shape is invalid"));
    }
    Ok(k_ps_uw
        .iter()
        .enumerate()
        .flat_map(|(row, &k)| {
            bperp_mat[row * n_ifg..(row + 1) * n_ifg]
                .iter()
                .map(move |&baseline| (k * baseline) as f32)
        })
        .collect())
}

fn update_bounds<T: Bounds>(minimum: &mut T, maximum: &mut T, value: T) {
    if value.is_nan() || minimum.is_nan() || maximum.is_nan() {
        *minimum = T::nan();
        *maximum = T::nan();
    } else {
        *minimum = minimum.min(value);
        *maximum = maximum.max(value);
    }
}

trait Bounds: Copy {
    fn is_nan(self) -> bool;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn nan() -> Self;
}

impl Bounds for f64 {
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    fn nan() -> Self {
        f64::NAN
    }
}

impl Bounds for f32 {
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    fn nan() -> Self {
        f32::NAN
    }
}
