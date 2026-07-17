use super::types::{Stage4Config, Stage4Error, Stage4Input, Stage4Measurements, Stage4Output};

pub fn weed_stage4(
    input: &Stage4Input,
    measurements: &Stage4Measurements,
    config: &Stage4Config,
) -> Result<Stage4Output, Stage4Error> {
    if input.selected_ix.len() != input.selection_keep.len() {
        return Err(Stage4Error::InvalidInput("selection mask length mismatch"));
    }
    let selected_ix = input
        .selected_ix
        .iter()
        .zip(&input.selection_keep)
        .filter_map(|(&index, &keep)| keep.then_some(index))
        .collect::<Vec<_>>();
    let n = selected_ix.len();
    if measurements.adjacency_keep.len() != n || measurements.duplicate_keep.len() != n {
        return Err(Stage4Error::InvalidInput("spatial mask length mismatch"));
    }
    if let Some(height) = &input.height {
        let max_source = selected_ix.iter().copied().max().unwrap_or(0);
        if max_source > height.len() {
            return Err(Stage4Error::InvalidInput(
                "height does not cover selected indices",
            ));
        }
    }

    let mut ix_weed = vec![true; n];
    for row in 0..n {
        ix_weed[row] &= measurements.adjacency_keep[row];
        ix_weed[row] &= measurements.duplicate_keep[row];
        if config.weed_zero_elevation {
            let height = input.height.as_ref().ok_or(Stage4Error::InvalidInput(
                "height is required for zero-elevation weeding",
            ))?;
            ix_weed[row] &= height[selected_ix[row] - 1] >= 1e-6;
        }
    }
    let pre_noise = ix_weed.iter().filter(|&&keep| keep).count();
    if measurements.ps_std.len() != pre_noise || measurements.ps_max.len() != pre_noise {
        return Err(Stage4Error::InvalidInput(
            "noise statistics length mismatch",
        ));
    }
    let ix_weed2 = measurements
        .ps_std
        .iter()
        .zip(&measurements.ps_max)
        .map(|(&std, &max)| std < config.weed_standard_dev && max < config.weed_max_noise)
        .collect::<Vec<_>>();
    let mut noise_row = 0usize;
    for keep in &mut ix_weed {
        if *keep {
            *keep = ix_weed2[noise_row];
            noise_row += 1;
        }
    }
    Ok(Stage4Output {
        selected_ix,
        ix_weed,
        ix_weed2,
        ps_std: measurements
            .ps_std
            .iter()
            .map(|&value| value as f32)
            .collect(),
        ps_max: measurements
            .ps_max
            .iter()
            .map(|&value| value as f32)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combines_selection_spatial_height_and_strict_noise_masks() {
        let input = Stage4Input {
            selected_ix: vec![1, 2, 3, 4],
            selection_keep: vec![true, false, true, true],
            height: Some(vec![10.0, 10.0, 0.0, 10.0]),
        };
        let measurements = Stage4Measurements {
            adjacency_keep: vec![true, true, true],
            duplicate_keep: vec![true, true, true],
            ps_std: vec![1.0, 0.2],
            ps_max: vec![0.2, 0.2],
        };
        let output = weed_stage4(
            &input,
            &measurements,
            &Stage4Config {
                weed_zero_elevation: true,
                weed_standard_dev: 1.0,
                weed_max_noise: 1.0,
            },
        )
        .unwrap();
        assert_eq!(output.selected_ix, vec![1, 3, 4]);
        assert_eq!(output.ix_weed, vec![false, false, true]);
    }
}
