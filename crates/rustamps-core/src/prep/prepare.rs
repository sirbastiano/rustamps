use super::candidates::candidate_statistics;
use super::patches::patch_grid;
use super::types::{
    MtPrepOptions, MtPrepOutput, PatchCandidate, PrepError, PreparedPatch, RasterShape,
    SnapPrepInput,
};

fn crop(
    values: &[f32],
    source: RasterShape,
    rows: super::types::InclusiveRange,
    columns: super::types::InclusiveRange,
) -> Vec<f32> {
    let mut output =
        Vec::with_capacity((rows.end - rows.start + 1) * (columns.end - columns.start + 1));
    for one_based_row in rows.start..=rows.end {
        let row = one_based_row - 1;
        let start = row * source.width + columns.start - 1;
        output.extend_from_slice(&values[start..start + columns.end - columns.start + 1]);
    }
    output
}

pub fn prepare_snap(
    input: &SnapPrepInput<'_>,
    options: MtPrepOptions,
) -> Result<MtPrepOutput, PrepError> {
    if !options.amp_dispersion.is_finite() || options.amp_dispersion < 0.0 {
        return Err(PrepError::InvalidOption("amp_dispersion"));
    }
    let stats = candidate_statistics(input, options.amp_dispersion)?;
    let grid = patch_grid(
        input.shape,
        options.range_patches,
        options.azimuth_patches,
        options.range_overlap,
        options.azimuth_overlap,
    )?;
    let mut patches = Vec::new();
    let mut candidate_count = 0usize;
    for (patch_index, bounds, no_overlap) in grid {
        let mut candidates = Vec::new();
        for source_index in 0..stats.selected.len() {
            if !stats.selected[source_index] {
                continue;
            }
            let row = source_index / input.shape.width;
            let column = source_index % input.shape.width;
            if !bounds.rows.contains(row + 1) || !bounds.columns.contains(column + 1) {
                continue;
            }
            let phase = input
                .diff_phase
                .iter()
                .map(|interferogram| interferogram[source_index])
                .collect();
            candidates.push(PatchCandidate {
                source_index,
                row,
                column,
                lon: input.lon[source_index],
                lat: input.lat[source_index],
                height: input.height[source_index],
                amplitude_dispersion: stats.amplitude_dispersion[source_index],
                phase,
            });
        }
        if candidates.is_empty() {
            continue;
        }
        candidate_count += candidates.len();
        patches.push(PreparedPatch {
            name: format!("PATCH_{patch_index}"),
            bounds,
            no_overlap,
            candidates,
            mean_amplitude: crop(
                &stats.normalized_amplitude_sum,
                input.shape,
                bounds.rows,
                bounds.columns,
            ),
            mean_amplitude_shape: RasterShape {
                length: bounds.rows.end - bounds.rows.start + 1,
                width: bounds.columns.end - bounds.columns.start + 1,
            },
        });
    }
    if patches.is_empty() {
        return Err(PrepError::NoCandidates);
    }
    Ok(MtPrepOutput {
        patches,
        candidate_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prep::types::{Complex32, SnapPrepInput};

    #[test]
    fn prepares_candidate_records_and_patch_crop() {
        let amplitudes = vec![10.0; 6];
        let amplitude_refs = vec![amplitudes.as_slice()];
        let phase = (0..6)
            .map(|index| Complex32::new(index as f32, -(index as f32)))
            .collect::<Vec<_>>();
        let phase_refs = vec![phase.as_slice()];
        let lon = vec![1.0; 6];
        let lat = vec![2.0; 6];
        let height = vec![3.0; 6];
        let input = SnapPrepInput {
            shape: RasterShape {
                length: 2,
                width: 3,
            },
            rslc_amplitudes: &amplitude_refs,
            diff_phase: &phase_refs,
            lon: &lon,
            lat: &lat,
            height: &height,
        };
        let output = prepare_snap(
            &input,
            MtPrepOptions {
                range_patches: 2,
                azimuth_patches: 1,
                range_overlap: 0,
                azimuth_overlap: 0,
                ..MtPrepOptions::default()
            },
        )
        .unwrap();
        assert_eq!(output.patches.len(), 2);
        assert_eq!(output.candidate_count, 6);
        assert_eq!(output.patches[0].candidates.len(), 2);
        assert_eq!(output.patches[1].candidates.len(), 4);
        assert_eq!(output.patches[0].mean_amplitude_shape.width, 1);
        assert_eq!(output.patches[1].candidates[0].source_index, 1);
    }
}
