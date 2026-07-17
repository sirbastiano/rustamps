use num_complex::Complex64;

#[derive(Clone, Copy)]
pub(super) struct AffineMoments {
    s0: f64,
    s1: f64,
    s2: f64,
    determinant: f64,
}

pub(super) fn affine_moments(x: &[f64], weights: &[f64]) -> AffineMoments {
    let s0 = weights.iter().sum();
    let s1 = weights
        .iter()
        .zip(x)
        .map(|(weight, value)| weight * value)
        .sum();
    let s2 = weights
        .iter()
        .zip(x)
        .map(|(weight, value)| weight * value * value)
        .sum();
    AffineMoments {
        s0,
        s1,
        s2,
        determinant: s0 * s2 - s1 * s1,
    }
}

pub(super) fn affine_one(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    moments: AffineMoments,
) -> (f64, f64) {
    let wy0 = y
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>();
    let wy1 = y
        .iter()
        .zip(weights)
        .zip(x)
        .map(|((value, weight), x)| value * weight * x)
        .sum::<f64>();
    if moments.determinant == 0.0 {
        let intercept = if moments.s0 == 0.0 {
            0.0
        } else {
            wy0 / moments.s0
        };
        (intercept, 0.0)
    } else {
        (
            (wy0 * moments.s2 - wy1 * moments.s1) / moments.determinant,
            (wy1 * moments.s0 - wy0 * moments.s1) / moments.determinant,
        )
    }
}

pub(super) struct EdgeScratch {
    pub smooth: Vec<Complex64>,
    pub adjusted: Vec<f64>,
    pub detrended: Vec<f64>,
}
