use num_complex::Complex32;
use pystamps_io::{MatArray, MatFile, MatValue};

pub fn scalar(value: f64) -> MatValue {
    MatValue::F64(MatArray {
        shape: vec![1, 1],
        values: vec![value],
    })
}

pub fn f64_array(shape: Vec<usize>, values: Vec<f64>) -> MatValue {
    MatValue::F64(MatArray { shape, values })
}

pub fn f32_array(shape: Vec<usize>, values: Vec<f32>) -> MatValue {
    MatValue::F32(MatArray { shape, values })
}

pub fn complex32_array(shape: Vec<usize>, values: Vec<Complex32>) -> MatValue {
    MatValue::ComplexF32(MatArray { shape, values })
}

pub fn numeric_f64(file: &MatFile, key: &str) -> Result<Vec<f64>, String> {
    let value = file
        .get(key)
        .ok_or_else(|| format!("missing MAT key {key}"))?;
    match value {
        MatValue::F64(v) => Ok(v.values.clone()),
        MatValue::F32(v) => Ok(v.values.iter().map(|&x| f64::from(x)).collect()),
        MatValue::I64(v) => Ok(v.values.iter().map(|&x| x as f64).collect()),
        MatValue::I32(v) => Ok(v.values.iter().map(|&x| f64::from(x)).collect()),
        MatValue::I16(v) => Ok(v.values.iter().map(|&x| f64::from(x)).collect()),
        MatValue::I8(v) => Ok(v.values.iter().map(|&x| f64::from(x)).collect()),
        MatValue::U64(v) => Ok(v.values.iter().map(|&x| x as f64).collect()),
        MatValue::U32(v) => Ok(v.values.iter().map(|&x| f64::from(x)).collect()),
        MatValue::U16(v) => Ok(v.values.iter().map(|&x| f64::from(x)).collect()),
        MatValue::U8(v) => Ok(v.values.iter().map(|&x| f64::from(x)).collect()),
        _ => Err(format!("MAT key {key} is not a real numeric array")),
    }
}

pub fn numeric_f32(file: &MatFile, key: &str) -> Result<Vec<f32>, String> {
    numeric_f64(file, key).map(|values| values.into_iter().map(|x| x as f32).collect())
}

pub fn scalar_f64(file: &MatFile, key: &str) -> Result<f64, String> {
    numeric_f64(file, key)?
        .first()
        .copied()
        .ok_or_else(|| format!("MAT key {key} is empty"))
}

pub fn bools(file: &MatFile, key: &str) -> Result<Vec<bool>, String> {
    match file
        .get(key)
        .ok_or_else(|| format!("missing MAT key {key}"))?
    {
        MatValue::Bool(value) => Ok(value.values.clone()),
        MatValue::U8(value) => Ok(value.values.iter().map(|&item| item != 0).collect()),
        MatValue::I8(value) => Ok(value.values.iter().map(|&item| item != 0).collect()),
        _ => numeric_f64(file, key)
            .map(|values| values.into_iter().map(|item| item != 0.0).collect()),
    }
}

pub fn shape(file: &MatFile, key: &str) -> Result<Vec<usize>, String> {
    file.get(key)
        .and_then(MatValue::shape)
        .map(<[usize]>::to_vec)
        .ok_or_else(|| format!("MAT key {key} has no dense shape"))
}

pub fn complex32(file: &MatFile, key: &str) -> Result<Vec<Complex32>, String> {
    match file
        .get(key)
        .ok_or_else(|| format!("missing MAT key {key}"))?
    {
        MatValue::ComplexF32(v) => Ok(v.values.clone()),
        MatValue::ComplexF64(v) => Ok(v
            .values
            .iter()
            .map(|x| Complex32::new(x.re as f32, x.im as f32))
            .collect()),
        MatValue::F32(v) => Ok(v.values.iter().map(|&x| Complex32::new(x, 0.0)).collect()),
        MatValue::F64(v) => Ok(v
            .values
            .iter()
            .map(|&x| Complex32::new(x as f32, 0.0))
            .collect()),
        _ => Err(format!("MAT key {key} is not a complex numeric array")),
    }
}
