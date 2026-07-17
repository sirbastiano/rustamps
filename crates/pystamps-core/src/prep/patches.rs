use super::types::{InclusiveRange, PatchBounds, PrepError, RasterShape};

pub fn patch_ranges(
    size: usize,
    count: usize,
    overlap: usize,
) -> Result<Vec<(InclusiveRange, InclusiveRange)>, PrepError> {
    if size == 0 {
        return Err(PrepError::InvalidOption("patch axis size must be positive"));
    }
    if count == 0 {
        return Err(PrepError::InvalidOption("patch count must be positive"));
    }
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let no_start = 1 + size.saturating_mul(index) / count;
        let mut no_end = size.saturating_mul(index + 1) / count;
        if index + 1 == count {
            no_end = size;
        }
        let patch_start = no_start.saturating_sub(overlap).max(1);
        let patch_end = no_end.saturating_add(overlap).min(size);
        output.push((
            InclusiveRange {
                start: patch_start,
                end: patch_end,
            },
            InclusiveRange {
                start: no_start,
                end: no_end,
            },
        ));
    }
    Ok(output)
}

pub fn patch_grid(
    shape: RasterShape,
    range_patches: usize,
    azimuth_patches: usize,
    range_overlap: usize,
    azimuth_overlap: usize,
) -> Result<Vec<(usize, PatchBounds, PatchBounds)>, PrepError> {
    shape.cells()?;
    let columns = patch_ranges(shape.width, range_patches, range_overlap)?;
    let rows = patch_ranges(shape.length, azimuth_patches, azimuth_overlap)?;
    let mut output = Vec::with_capacity(columns.len() * rows.len());
    let mut patch_index = 1usize;
    for (column_bounds, column_no_overlap) in columns {
        for &(row_bounds, row_no_overlap) in &rows {
            output.push((
                patch_index,
                PatchBounds {
                    columns: column_bounds,
                    rows: row_bounds,
                },
                PatchBounds {
                    columns: column_no_overlap,
                    rows: row_no_overlap,
                },
            ));
            patch_index += 1;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_match_mt_prep_one_based_overlap_contract() {
        let ranges = patch_ranges(10, 3, 2).unwrap();
        assert_eq!(
            ranges,
            vec![
                (
                    InclusiveRange { start: 1, end: 5 },
                    InclusiveRange { start: 1, end: 3 }
                ),
                (
                    InclusiveRange { start: 2, end: 8 },
                    InclusiveRange { start: 4, end: 6 }
                ),
                (
                    InclusiveRange { start: 5, end: 10 },
                    InclusiveRange { start: 7, end: 10 }
                ),
            ]
        );
    }

    #[test]
    fn grid_numbers_patches_column_then_row() {
        let grid = patch_grid(
            RasterShape {
                length: 4,
                width: 6,
            },
            2,
            2,
            0,
            0,
        )
        .unwrap();
        assert_eq!(grid.len(), 4);
        assert_eq!(grid[0].0, 1);
        assert_eq!(grid[0].1.columns, InclusiveRange { start: 1, end: 3 });
        assert_eq!(grid[0].1.rows, InclusiveRange { start: 1, end: 2 });
        assert_eq!(grid[1].1.columns, InclusiveRange { start: 1, end: 3 });
        assert_eq!(grid[1].1.rows, InclusiveRange { start: 3, end: 4 });
        assert_eq!(grid[2].1.columns, InclusiveRange { start: 4, end: 6 });
    }
}
