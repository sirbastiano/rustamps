use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MatrixError {
    #[error("matrix shape {rows}x{cols} requires {expected} values, received {actual}")]
    Shape {
        rows: usize,
        cols: usize,
        expected: usize,
        actual: usize,
    },
    #[error("matrix dimensions overflow usize")]
    DimensionOverflow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    values: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn from_vec(rows: usize, cols: usize, values: Vec<T>) -> Result<Self, MatrixError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(MatrixError::DimensionOverflow)?;
        if expected != values.len() {
            return Err(MatrixError::Shape {
                rows,
                cols,
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { rows, cols, values })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    pub fn into_vec(self) -> Vec<T> {
        self.values
    }

    pub fn row(&self, row: usize) -> &[T] {
        let start = row * self.cols;
        &self.values[start..start + self.cols]
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        let start = row * self.cols;
        &mut self.values[start..start + self.cols]
    }
}

impl<T: Clone> Matrix<T> {
    pub fn filled(rows: usize, cols: usize, value: T) -> Result<Self, MatrixError> {
        let len = rows
            .checked_mul(cols)
            .ok_or(MatrixError::DimensionOverflow)?;
        Ok(Self {
            rows,
            cols,
            values: vec![value; len],
        })
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.values[row * self.cols + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.values[row * self.cols + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_indexing_is_explicit() {
        let matrix = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(matrix[(1, 2)], 6);
        assert_eq!(matrix.row(1), &[4, 5, 6]);
    }

    #[test]
    fn rejects_shape_mismatch() {
        assert!(matches!(
            Matrix::from_vec(2, 2, vec![1, 2, 3]),
            Err(MatrixError::Shape { .. })
        ));
    }
}
