use std::collections::BTreeMap;

use num_complex::{Complex32, Complex64};

#[derive(Debug, Clone, PartialEq)]
pub struct MatArray<T> {
    pub shape: Vec<usize>,
    pub values: Vec<T>,
}

impl<T> MatArray<T> {
    pub fn new(shape: Vec<usize>, values: Vec<T>) -> Option<Self> {
        let expected = shape
            .iter()
            .try_fold(1_usize, |acc, &value| acc.checked_mul(value))?;
        (expected == values.len()).then_some(Self { shape, values })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatSparse {
    pub rows: usize,
    pub cols: usize,
    pub row_indices: Vec<usize>,
    pub column_offsets: Vec<usize>,
    pub real: Vec<f64>,
    pub imag: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatValue {
    F64(MatArray<f64>),
    F32(MatArray<f32>),
    I64(MatArray<i64>),
    I32(MatArray<i32>),
    I16(MatArray<i16>),
    I8(MatArray<i8>),
    U64(MatArray<u64>),
    U32(MatArray<u32>),
    U16(MatArray<u16>),
    U8(MatArray<u8>),
    Bool(MatArray<bool>),
    ComplexF64(MatArray<Complex64>),
    ComplexF32(MatArray<Complex32>),
    Char(MatArray<u16>),
    Sparse(MatSparse),
    Cell(Vec<MatValue>),
    Struct(BTreeMap<String, MatValue>),
}

impl MatValue {
    pub fn shape(&self) -> Option<&[usize]> {
        match self {
            Self::F64(v) => Some(&v.shape),
            Self::F32(v) => Some(&v.shape),
            Self::I64(v) => Some(&v.shape),
            Self::I32(v) => Some(&v.shape),
            Self::I16(v) => Some(&v.shape),
            Self::I8(v) => Some(&v.shape),
            Self::U64(v) => Some(&v.shape),
            Self::U32(v) => Some(&v.shape),
            Self::U16(v) => Some(&v.shape),
            Self::U8(v) => Some(&v.shape),
            Self::Bool(v) => Some(&v.shape),
            Self::ComplexF64(v) => Some(&v.shape),
            Self::ComplexF32(v) => Some(&v.shape),
            Self::Char(v) => Some(&v.shape),
            Self::Sparse(_) | Self::Cell(_) | Self::Struct(_) => None,
        }
    }
}

pub type MatFile = BTreeMap<String, MatValue>;
