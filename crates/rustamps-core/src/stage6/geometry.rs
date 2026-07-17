use super::Stage6Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IfgSets {
    pub unwrap_indices: Vec<usize>,
    pub solve_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleMasterGeometry {
    pub unwrap_indices: Vec<usize>,
    pub ifgday_pairs: Vec<[usize; 2]>,
}

pub fn unwrap_ifg_sets(
    n_ifg: usize,
    master_index: usize,
    drop_indices: &[usize],
    small_baseline: bool,
) -> Result<IfgSets, Stage6Error> {
    if !small_baseline && master_index >= n_ifg {
        return Err(Stage6Error::new(
            "master_index must be within the interferogram stack",
        ));
    }
    if drop_indices.iter().any(|&index| index >= n_ifg) {
        return Err(Stage6Error::new(
            "drop_indices contains an out-of-range interferogram",
        ));
    }
    let mut dropped = vec![false; n_ifg];
    for &index in drop_indices {
        dropped[index] = true;
    }
    let unwrap_indices = (0..n_ifg)
        .filter(|&index| !dropped[index])
        .collect::<Vec<_>>();
    let solve_indices = unwrap_indices
        .iter()
        .copied()
        .filter(|&index| small_baseline || index != master_index)
        .collect();
    Ok(IfgSets {
        unwrap_indices,
        solve_indices,
    })
}

pub fn single_master_ifg_geometry(
    n_ifg: usize,
    master_index: usize,
) -> Result<SingleMasterGeometry, Stage6Error> {
    if master_index >= n_ifg {
        return Err(Stage6Error::new(
            "master_index must be within the interferogram stack",
        ));
    }
    let unwrap_indices = (0..n_ifg)
        .filter(|&index| index != master_index)
        .collect::<Vec<_>>();
    let ifgday_pairs = unwrap_indices
        .iter()
        .map(|&index| [master_index, index])
        .collect();
    Ok(SingleMasterGeometry {
        unwrap_indices,
        ifgday_pairs,
    })
}
