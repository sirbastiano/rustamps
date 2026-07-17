use crate::stage6::unwrap::native::{horizontal_index, rounded_delta, vertical_index, EdgeDatum};

pub(crate) fn edge_residues(
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> Vec<i32> {
    let prn = nrow.saturating_sub(1);
    let pcn = ncol.saturating_sub(1);
    let mut residues = vec![0; prn * pcn];
    for row in 0..prn {
        for col in 0..pcn {
            let (Some(top), Some(right), Some(bottom), Some(left)) = (
                horizontal[horizontal_index(row, col, ncol)],
                vertical[vertical_index(row, col + 1, ncol)],
                horizontal[horizontal_index(row + 1, col, ncol)],
                vertical[vertical_index(row, col, ncol)],
            ) else {
                continue;
            };
            residues[row * pcn + col] = rounded_delta(top) + rounded_delta(right)
                - rounded_delta(bottom)
                - rounded_delta(left);
        }
    }
    residues
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(delta: f32) -> EdgeDatum {
        EdgeDatum {
            cost: 1,
            desired_delta: delta,
            offset: 0,
            dzmax: 32000,
            laycost: -32000,
            nshortcycle: 200,
            flow_sign: 1,
            flow: 0,
        }
    }

    #[test]
    fn edge_residues_match_wrapped_plaquette_curl_signs() {
        let nrow = 2;
        let ncol = 2;
        let horizontal = vec![Some(edge(1.0)), Some(edge(0.0))];
        let vertical = vec![Some(edge(0.0)), Some(edge(0.0))];

        assert_eq!(edge_residues(&horizontal, &vertical, nrow, ncol), vec![1]);
    }

    #[test]
    fn edge_residues_skip_plaquettes_with_missing_arcs() {
        let nrow = 2;
        let ncol = 2;
        let horizontal = vec![Some(edge(1.0)), None];
        let vertical = vec![Some(edge(0.0)), Some(edge(0.0))];

        assert_eq!(edge_residues(&horizontal, &vertical, nrow, ncol), vec![0]);
    }
}
