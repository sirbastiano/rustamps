use super::types::{NoOverlapBounds, PromotedPatch, Stage5Error, Stage5Merged, Stage5Row};
use crate::stages::stage1::{local_xy, quantize_millimeters, Matrix};
use std::collections::HashMap;

fn key(row: &Stage5Row) -> (i64, i64) {
    (row.ij[1].round() as i64, row.ij[2].round() as i64)
}

fn in_no_overlap(row: &Stage5Row, bounds: NoOverlapBounds) -> bool {
    let first = row.ij[1].round() as i64;
    let second = row.ij[2].round() as i64;
    first >= bounds.column_min - 1
        && first < bounds.column_max
        && second >= bounds.row_min - 1
        && second < bounds.row_max
}

pub fn merge_patches(
    patches: &[PromotedPatch],
    merge_resample_size: f64,
) -> Result<Stage5Merged, Stage5Error> {
    merge_patches_with_heading(patches, merge_resample_size, None)
}

pub fn merge_patches_with_heading(
    patches: &[PromotedPatch],
    merge_resample_size: f64,
    heading_deg: Option<f64>,
) -> Result<Stage5Merged, Stage5Error> {
    if merge_resample_size != 0.0 {
        return Err(Stage5Error::UnsupportedResampling);
    }
    let mut merged: Vec<Option<Stage5Row>> = Vec::new();
    let mut by_pixel: HashMap<(i64, i64), usize> = HashMap::new();
    for patch in patches {
        for row in &patch.rows {
            let pixel = key(row);
            let owns = patch
                .no_overlap
                .is_none_or(|bounds| in_no_overlap(row, bounds));
            if let Some(&prior) = by_pixel.get(&pixel) {
                if !owns {
                    continue;
                }
                merged[prior] = None;
            }
            let next = merged.len();
            merged.push(Some(row.clone()));
            by_pixel.insert(pixel, next);
        }
    }
    let mut rows = merged.into_iter().flatten().collect::<Vec<_>>();
    let mut best_lonlat: HashMap<(u64, u64), usize> = HashMap::new();
    let mut keep = vec![true; rows.len()];
    for index in 0..rows.len() {
        let lonlat = (
            rows[index].lonlat[0].to_bits(),
            rows[index].lonlat[1].to_bits(),
        );
        if let Some(&prior) = best_lonlat.get(&lonlat) {
            if rows[index].coherence > rows[prior].coherence {
                keep[prior] = false;
                best_lonlat.insert(lonlat, index);
            } else {
                keep[index] = false;
            }
        } else {
            best_lonlat.insert(lonlat, index);
        }
    }
    rows = rows
        .into_iter()
        .zip(keep)
        .filter_map(|(row, keep)| keep.then_some(row))
        .collect();
    if rows.is_empty() {
        return Ok(Stage5Merged {
            rows,
            xy: Matrix {
                rows: 0,
                cols: 3,
                values: Vec::new(),
            },
            xy_origin: [0.0, 0.0],
        });
    }
    let lonlat = rows.iter().map(|row| row.lonlat).collect::<Vec<_>>();
    let (local, xy_origin) = local_xy(&lonlat, heading_deg)
        .map_err(|_| Stage5Error::InvalidInput("merged lonlat cannot be projected"))?;
    let local_f32 = local
        .iter()
        .map(|value| [value[0] as f32, value[1] as f32])
        .collect::<Vec<_>>();
    let mut order = (0..rows.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        local_f32[left][1]
            .total_cmp(&local_f32[right][1])
            .then_with(|| local_f32[left][0].total_cmp(&local_f32[right][0]))
    });
    rows = order.iter().map(|&index| rows[index].clone()).collect();
    let mut xy_values = Vec::with_capacity(rows.len() * 3);
    for (index, &source) in order.iter().enumerate() {
        xy_values.push((index + 1) as f32);
        xy_values.push(quantize_millimeters(f64::from(local_f32[source][0])));
        xy_values.push(quantize_millimeters(f64::from(local_f32[source][1])));
    }
    for (index, row) in rows.iter_mut().enumerate() {
        row.ij[0] = (index + 1) as f64;
    }
    Ok(Stage5Merged {
        xy: Matrix {
            rows: rows.len(),
            cols: 3,
            values: xy_values,
        },
        rows,
        xy_origin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::stage1::Complex32;

    fn row(first: f64, second: f64, coherence: f64) -> Stage5Row {
        Stage5Row {
            ij: [1.0, first, second],
            lonlat: [first, second],
            phase: vec![Complex32::new(1.0, 0.0)],
            k_ps: 0.0,
            c_ps: 0.0,
            coherence,
            phase_patch: vec![],
            phase_residual: vec![],
            bperp: None,
            height: None,
            look_angle: None,
            amplitude_dispersion: None,
        }
    }

    #[test]
    fn later_patch_replaces_overlap_only_inside_its_core() {
        let first = PromotedPatch {
            name: "PATCH_1".into(),
            no_overlap: None,
            rows: vec![row(10.0, 20.0, 0.5)],
        };
        let second = PromotedPatch {
            name: "PATCH_2".into(),
            no_overlap: Some(NoOverlapBounds {
                row_min: 21,
                row_max: 21,
                column_min: 11,
                column_max: 11,
            }),
            rows: vec![row(10.0, 20.0, 0.9)],
        };
        let output = merge_patches(&[first, second], 0.0).unwrap();
        assert_eq!(output.rows.len(), 1);
        assert_eq!(output.rows[0].coherence, 0.9);
        assert_eq!(output.xy.rows, 1);
    }
}
