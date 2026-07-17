use std::path::{Path, PathBuf};

use rustamps_io::{discover_dataset, DatasetError};
use serde::{Deserialize, Serialize};

const PATCH_ARTIFACTS: [(u8, &str); 5] = [
    (1, "ps1.mat"),
    (2, "pm1.mat"),
    (3, "select1.mat"),
    (4, "weed1.mat"),
    (5, "ph2.mat"),
];
const MERGED_ARTIFACTS: [(u8, &str); 4] = [
    (5, "ifgstd2.mat"),
    (6, "phuw2.mat"),
    (7, "scla2.mat"),
    (8, "scn2.mat"),
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatchStatus {
    pub patch: String,
    pub stage: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetStatus {
    pub dataset: PathBuf,
    pub merged_stage: u8,
    pub patches: Vec<PatchStatus>,
}

pub fn collect_status(root: impl AsRef<Path>) -> Result<DatasetStatus, DatasetError> {
    let layout = discover_dataset(root)?;
    let patches = layout
        .patches
        .iter()
        .map(|patch| PatchStatus {
            patch: patch
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned(),
            stage: infer_stage(patch, &PATCH_ARTIFACTS),
        })
        .collect();
    Ok(DatasetStatus {
        dataset: layout.root.clone(),
        merged_stage: infer_stage(&layout.root, &MERGED_ARTIFACTS),
        patches,
    })
}

fn infer_stage(root: &Path, artifacts: &[(u8, &str)]) -> u8 {
    artifacts
        .iter()
        .take_while(|(_, artifact)| root.join(artifact).is_file())
        .map(|(stage, _)| *stage)
        .last()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("clock must follow Unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rustamps-status-{label}-{nonce}"));
        fs::create_dir(&root).expect("create temporary directory");
        root
    }

    #[test]
    fn merged_artifacts_report_their_pipeline_stage() {
        let root = temp_dir("complete");
        for (_, artifact) in MERGED_ARTIFACTS {
            fs::write(root.join(artifact), []).expect("create artifact");
        }

        assert_eq!(infer_stage(&root, &MERGED_ARTIFACTS), 8);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }

    #[test]
    fn stage_inference_stops_at_the_first_missing_artifact() {
        let root = temp_dir("gap");
        fs::write(root.join("ifgstd2.mat"), []).expect("create stage 5 artifact");
        fs::write(root.join("scla2.mat"), []).expect("create stale stage 7 artifact");

        assert_eq!(infer_stage(&root, &MERGED_ARTIFACTS), 5);
        fs::remove_dir_all(root).expect("remove temporary directory");
    }
}
