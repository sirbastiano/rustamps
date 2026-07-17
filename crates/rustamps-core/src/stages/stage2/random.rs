use super::{histogram_with_centers, topofit_batch, Stage2Error};
use crate::stages::stage1::{Complex32, Matrix};

pub fn random_coherence_histogram(
    seed: u32,
    sample_count: usize,
    bperp: &[f64],
    n_trial_wraps: f64,
    centers: &[f64],
) -> Result<Vec<f64>, Stage2Error> {
    if sample_count == 0 || bperp.is_empty() {
        return Err(Stage2Error::InvalidInput(
            "random histogram requires samples and baselines",
        ));
    }
    let mut rng = MatlabV5Rng::new(seed);
    let mut uniforms = vec![0.0; sample_count * bperp.len()];
    uniforms.iter_mut().for_each(|value| *value = rng.uniform());
    let mut phase = Vec::with_capacity(uniforms.len());
    for row in 0..sample_count {
        for col in 0..bperp.len() {
            let angle = uniforms[col * sample_count + row] * 2.0 * std::f64::consts::PI;
            phase.push(Complex32::new(angle.cos() as f32, angle.sin() as f32));
        }
    }
    let result = topofit_batch(
        &Matrix {
            rows: sample_count,
            cols: bperp.len(),
            values: phase,
        },
        &Matrix {
            rows: sample_count,
            cols: bperp.len(),
            values: bperp.repeat(sample_count),
        },
        n_trial_wraps,
    )?;
    Ok(histogram_with_centers(&result.coherence, centers))
}

struct MatlabV5Rng {
    state: [f64; 32],
    index: usize,
    borrow: f64,
    state_bits: u32,
}

impl MatlabV5Rng {
    fn new(seed: u32) -> Self {
        let mut rng = Self {
            state: [0.0; 32],
            index: 0,
            borrow: 0.0,
            state_bits: if seed == 0 { 1 << 31 } else { seed },
        };
        let mut setup_bits = rng.state_bits;
        for index in 0..32 {
            let mut bits = 0_u64;
            for _ in 0..53 {
                setup_bits = xorshift(setup_bits);
                bits = (bits << 1) | u64::from((setup_bits >> 19) & 1);
            }
            rng.state[index] = (bits as f64) * 2_f64.powi(-53);
        }
        rng
    }

    fn uniform(&mut self) -> f64 {
        let mut value =
            self.state[(self.index + 20) & 31] - self.state[(self.index + 5) & 31] - self.borrow;
        if value < 0.0 {
            value += 1.0;
            self.borrow = 2_f64.powi(-53);
        } else {
            self.borrow = 0.0;
        }
        self.state[self.index] = value;
        self.index = (self.index + 1) & 31;
        let low = self.state_bits;
        self.state_bits = xorshift(low);
        let mask = (((self.state_bits as u64) << 32) & ((1_u64 << 52) - 1)) ^ u64::from(low);
        if value == 0.0 {
            return (mask as f64) * 2_f64.powi(-53);
        }
        let bits = value.to_bits();
        f64::from_bits((bits & !((1_u64 << 52) - 1)) | ((bits & ((1_u64 << 52) - 1)) ^ mask))
    }
}

fn xorshift(mut value: u32) -> u32 {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matlab_v5_seed_has_stable_first_values() {
        let mut rng = MatlabV5Rng::new(2005);
        let values = (0..3).map(|_| rng.uniform()).collect::<Vec<_>>();
        let expected = [
            0.092_958_990_583_191_2,
            0.373_737_758_407_522_77,
            0.448_770_579_371_177_65,
        ];
        for (actual, expected) in values.iter().zip(expected) {
            assert_eq!(*actual, expected);
        }
    }
}
