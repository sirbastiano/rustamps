use std::path::Path;

use rustamps_core::stages::stage4::Stage4Output;
use rustamps_io::{write_mat, MatArray, MatFile, MatValue, StageTransaction};

use super::super::mat::{f32_array, f64_array};

pub(super) fn write(patch: &Path, output: Stage4Output, ifg_index: &[f64]) -> Result<(), String> {
    let weed_count = output.ix_weed.len();
    let noise_count = output.ix_weed2.len();
    let mut payload = MatFile::new();
    payload.insert(
        "ifg_index".to_owned(),
        f64_array(vec![1, ifg_index.len()], ifg_index.to_vec()),
    );
    payload.insert(
        "ix_weed".to_owned(),
        MatValue::U8(MatArray {
            shape: vec![weed_count, 1],
            values: output.ix_weed.into_iter().map(u8::from).collect(),
        }),
    );
    payload.insert(
        "ix_weed2".to_owned(),
        MatValue::U8(MatArray {
            shape: vec![noise_count, 1],
            values: output.ix_weed2.into_iter().map(u8::from).collect(),
        }),
    );
    payload.insert(
        "ps_max".to_owned(),
        f32_array(vec![noise_count, 1], output.ps_max),
    );
    payload.insert(
        "ps_std".to_owned(),
        f32_array(vec![noise_count, 1], output.ps_std),
    );
    let transaction =
        StageTransaction::begin(patch, "stage4").map_err(|error| error.to_string())?;
    write_mat(transaction.path("weed1.mat"), &payload).map_err(|error| error.to_string())?;
    transaction
        .commit(&["weed1.mat"], "weed1.mat")
        .map_err(|error| error.to_string())
}
