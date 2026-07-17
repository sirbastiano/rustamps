use super::types::{PromotedPatch, Stage5Error, Stage5PatchInput, Stage5Row};

pub fn promote_patch(
    name: impl Into<String>,
    input: &Stage5PatchInput,
) -> Result<PromotedPatch, Stage5Error> {
    let n = input.ij.rows;
    if input.ij.cols != 3
        || input.lonlat.rows != n
        || input.lonlat.cols != 2
        || input.phase.rows != n
        || input.phase_patch.rows != n
        || input.phase_residual.rows != n
        || input.k_ps.len() != n
        || input.c_ps.len() != n
        || input.coherence.len() != n
        || input.retain.len() != n
    {
        return Err(Stage5Error::InvalidInput(
            "row-aligned arrays have incompatible shapes",
        ));
    }
    if input
        .bperp_mat
        .as_ref()
        .is_some_and(|matrix| matrix.rows != n)
        || input
            .height
            .as_ref()
            .is_some_and(|values| values.len() != n)
        || input
            .look_angle
            .as_ref()
            .is_some_and(|values| values.len() != n)
        || input
            .amplitude_dispersion
            .as_ref()
            .is_some_and(|values| values.len() != n)
    {
        return Err(Stage5Error::InvalidInput(
            "optional artifact row count mismatch",
        ));
    }
    let mut rows = Vec::new();
    for source in 0..n {
        if !input.retain[source] {
            continue;
        }
        rows.push(Stage5Row {
            ij: [
                input.ij.row(source)[0],
                input.ij.row(source)[1],
                input.ij.row(source)[2],
            ],
            lonlat: [input.lonlat.row(source)[0], input.lonlat.row(source)[1]],
            phase: input.phase.row(source).to_vec(),
            k_ps: input.k_ps[source],
            c_ps: input.c_ps[source],
            coherence: input.coherence[source],
            phase_patch: input.phase_patch.row(source).to_vec(),
            phase_residual: input.phase_residual.row(source).to_vec(),
            bperp: input
                .bperp_mat
                .as_ref()
                .map(|matrix| matrix.row(source).to_vec()),
            height: input.height.as_ref().map(|values| values[source]),
            look_angle: input.look_angle.as_ref().map(|values| values[source]),
            amplitude_dispersion: input
                .amplitude_dispersion
                .as_ref()
                .map(|values| values[source]),
        });
    }
    Ok(PromotedPatch {
        name: name.into(),
        no_overlap: None,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::stage1::{Complex32, Matrix};

    #[test]
    fn promotion_keeps_all_artifacts_aligned() {
        let input = Stage5PatchInput {
            ij: Matrix::new(2, 3, vec![1.0, 10.0, 20.0, 2.0, 30.0, 40.0]).unwrap(),
            lonlat: Matrix::new(2, 2, vec![12.0, 45.0, 13.0, 46.0]).unwrap(),
            phase: Matrix::new(2, 1, vec![Complex32::new(1.0, 0.0); 2]).unwrap(),
            k_ps: vec![1.0, 2.0],
            c_ps: vec![3.0, 4.0],
            coherence: vec![0.5, 0.9],
            phase_patch: Matrix::new(2, 1, vec![Complex32::new(1.0, 0.0); 2]).unwrap(),
            phase_residual: Matrix::new(2, 1, vec![0.0; 2]).unwrap(),
            retain: vec![false, true],
            bperp_mat: None,
            height: Some(vec![10.0, 20.0]),
            look_angle: None,
            amplitude_dispersion: None,
        };
        let output = promote_patch("PATCH_1", &input).unwrap();
        assert_eq!(output.rows.len(), 1);
        assert_eq!(output.rows[0].ij, [2.0, 30.0, 40.0]);
        assert_eq!(output.rows[0].height, Some(20.0));
    }
}
