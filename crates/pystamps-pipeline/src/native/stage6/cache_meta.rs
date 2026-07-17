use std::collections::BTreeMap;

use num_complex::Complex32;
use pystamps_io::{MatFile, MatValue};

use super::super::mat::scalar;

// Bump whenever grid, interpolation, or space-time cache semantics change.
const SCHEMA_VERSION: u64 = 1;
const HASH_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const HASH_PRIME: u64 = 0x0000_0100_0000_01b3;
const MATLAB_INTEGER_MASK: u64 = (1_u64 << 52) - 1;

pub fn insert(file: &mut MatFile, fingerprint: u64, checksum: u64) {
    for (key, value) in [
        ("pystamps_stage6_cache_schema", SCHEMA_VERSION),
        ("pystamps_input_fingerprint", fingerprint),
        ("payload_checksum", checksum),
    ] {
        file.insert(key.to_owned(), scalar(value as f64));
    }
}

pub fn matches(file: &MatFile, fingerprint: u64) -> Result<bool, String> {
    let Some(schema) = integer(file, "pystamps_stage6_cache_schema")? else {
        return Ok(false);
    };
    if schema != SCHEMA_VERSION {
        return Ok(false);
    }
    Ok(integer(file, "pystamps_input_fingerprint")? == Some(fingerprint))
}

pub fn validate(file: &MatFile, expected: u64, label: &str) -> Result<(), String> {
    let found = integer(file, "payload_checksum")?
        .ok_or_else(|| format!("{label} cache is missing payload_checksum"))?;
    if found == expected {
        Ok(())
    } else {
        Err(format!(
            "{label} payload checksum {found} does not match {expected}"
        ))
    }
}

fn integer(file: &BTreeMap<String, MatValue>, key: &str) -> Result<Option<u64>, String> {
    let Some(value) = file.get(key) else {
        return Ok(None);
    };
    let MatValue::F64(array) = value else {
        return Err(format!("{key} is not an f64 scalar"));
    };
    if array.shape != [1, 1] {
        return Err(format!("{key} is not an f64 scalar"));
    }
    let value = array.values[0];
    if !value.is_finite()
        || value < 0.0
        || value > MATLAB_INTEGER_MASK as f64
        || value.fract() != 0.0
    {
        return Err(format!("{key} is not an exact non-negative MAT integer"));
    }
    Ok(Some(value as u64))
}

pub struct Checksum(u64);

impl Checksum {
    pub fn new(fingerprint: u64) -> Self {
        let mut checksum = Self(HASH_OFFSET);
        checksum.u64(fingerprint);
        checksum
    }

    pub fn u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(HASH_PRIME);
        }
    }

    pub fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    pub fn f32(&mut self, value: f32) {
        self.u64(u64::from(value.to_bits()));
    }

    pub fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    pub fn bool(&mut self, value: bool) {
        self.u64(u64::from(value));
    }

    pub fn complex32(&mut self, value: Complex32) {
        self.f32(value.re);
        self.f32(value.im);
    }

    pub fn finish(self) -> u64 {
        (self.0 & MATLAB_INTEGER_MASK).max(1)
    }
}
