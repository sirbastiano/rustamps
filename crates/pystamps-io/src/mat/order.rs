pub(super) fn column_to_row<T: Clone>(values: Vec<T>, shape: &[usize]) -> Vec<T> {
    if values.len() <= 1 || shape.len() <= 1 {
        return values;
    }
    let mut output = values.clone();
    for row_index in 0..values.len() {
        let column_index = row_to_column_index(row_index, shape);
        output[row_index] = values[column_index].clone();
    }
    output
}

pub(super) fn row_to_column<T: Clone>(values: &[T], shape: &[usize]) -> Vec<T> {
    if values.len() <= 1 || shape.len() <= 1 {
        return values.to_vec();
    }
    let mut output = values.to_vec();
    for row_index in 0..values.len() {
        let column_index = row_to_column_index(row_index, shape);
        output[column_index] = values[row_index].clone();
    }
    output
}

fn row_to_column_index(mut row_index: usize, shape: &[usize]) -> usize {
    let mut coordinates = vec![0_usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = row_index % shape[axis];
        row_index /= shape[axis];
    }
    let mut stride = 1_usize;
    let mut column_index = 0_usize;
    for (&coordinate, &dimension) in coordinates.iter().zip(shape) {
        column_index += coordinate * stride;
        stride *= dimension;
    }
    column_index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_matlab_column_major_to_row_major() {
        let matlab = vec![1, 4, 2, 5, 3, 6];
        let row = column_to_row(matlab, &[2, 3]);
        assert_eq!(row, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(row_to_column(&row, &[2, 3]), vec![1, 4, 2, 5, 3, 6]);
    }
}
