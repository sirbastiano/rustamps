use std::path::{Path, PathBuf};

use rustamps_io::{discover_dataset, DatasetError};

const PATCH_ARTIFACTS: &[(u8, &str)] = &[
    (1, "ps1.mat"),
    (1, "ph1.mat"),
    (1, "bp1.mat"),
    (1, "da1.mat"),
    (1, "hgt1.mat"),
    (1, "la1.mat"),
    (1, "inc1.mat"),
    (2, "pm1.mat"),
    (3, "select1.mat"),
    (4, "weed1.mat"),
    (5, "ps2.mat"),
    (5, "ph2.mat"),
    (5, "pm2.mat"),
    (5, "bp2.mat"),
    (5, "rc2.mat"),
    (5, "da2.mat"),
    (5, "hgt2.mat"),
    (5, "la2.mat"),
    (5, "inc2.mat"),
    (5, "psver.mat"),
];
const ROOT_ARTIFACTS: &[(u8, &str, bool)] = &[
    (5, "ps2.mat", true),
    (5, "pm2.mat", true),
    (5, "ph2.mat", true),
    (5, "bp2.mat", true),
    (5, "rc2.mat", true),
    (5, "da2.mat", true),
    (5, "hgt2.mat", true),
    (5, "la2.mat", true),
    (5, "inc2.mat", true),
    (6, "phuw2.mat", true),
    (7, "scla2.mat", true),
    (8, "scn2.mat", true),
    (5, "ifgstd2.mat", true),
    (6, "uw_space_time.mat", false),
    (6, "uw_grid.mat", false),
    (6, "uw_interp.mat", false),
    (6, "uw_phaseuw.mat", false),
    (7, "scla_smooth2.mat", false),
    (5, "psver.mat", true),
];

pub(crate) fn artifact_paths(
    root: &Path,
    through_stage: Option<u8>,
    final_products_only: bool,
) -> Result<Vec<PathBuf>, DatasetError> {
    let layout = discover_dataset(root)?;
    let mut paths = Vec::new();
    for patch in layout.patches {
        let relative_patch = patch
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| patch.clone());
        for &(stage, name) in PATCH_ARTIFACTS {
            if included(stage, through_stage) && patch.join(name).is_file() {
                paths.push(relative_patch.join(name));
            }
        }
    }
    for &(stage, name, final_product) in ROOT_ARTIFACTS {
        if included(stage, through_stage)
            && (!final_products_only || final_product)
            && root.join(name).is_file()
        {
            paths.push(PathBuf::from(name));
        }
    }
    Ok(paths)
}

fn included(stage: u8, through_stage: Option<u8>) -> bool {
    through_stage.is_none_or(|maximum| stage <= maximum)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn discovers_patch_and_root_artifacts_in_stable_order() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rustamps-artifacts-{stamp}"));
        fs::create_dir_all(root.join("PATCH_2")).unwrap();
        fs::create_dir_all(root.join("PATCH_1")).unwrap();
        fs::write(root.join("PATCH_2/pm1.mat"), []).unwrap();
        fs::write(root.join("PATCH_1/ps1.mat"), []).unwrap();
        fs::write(root.join("PATCH_1/ph1.mat"), []).unwrap();
        fs::write(root.join("PATCH_1/bp1.mat"), []).unwrap();
        fs::write(root.join("PATCH_1/rc2.mat"), []).unwrap();
        fs::write(root.join("ps2.mat"), []).unwrap();
        fs::write(root.join("bp2.mat"), []).unwrap();
        fs::write(root.join("rc2.mat"), []).unwrap();

        let paths = artifact_paths(&root, None, false).unwrap();
        assert_eq!(
            paths,
            vec![
                PathBuf::from("PATCH_1/ps1.mat"),
                PathBuf::from("PATCH_1/ph1.mat"),
                PathBuf::from("PATCH_1/bp1.mat"),
                PathBuf::from("PATCH_1/rc2.mat"),
                PathBuf::from("PATCH_2/pm1.mat"),
                PathBuf::from("ps2.mat"),
                PathBuf::from("bp2.mat"),
                PathBuf::from("rc2.mat")
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn through_stage_excludes_later_and_nonproduction_artifacts() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rustamps-artifact-scope-{stamp}"));
        fs::create_dir_all(root.join("PATCH_1")).unwrap();
        for name in [
            "ps2.mat",
            "phuw2.mat",
            "scla2.mat",
            "scn2.mat",
            "mean_v.mat",
        ] {
            fs::write(root.join(name), []).unwrap();
        }

        assert_eq!(
            artifact_paths(&root, Some(6), false).unwrap(),
            vec![PathBuf::from("ps2.mat"), PathBuf::from("phuw2.mat")]
        );
        assert_eq!(
            artifact_paths(&root, None, false).unwrap(),
            vec![
                PathBuf::from("ps2.mat"),
                PathBuf::from("phuw2.mat"),
                PathBuf::from("scla2.mat"),
                PathBuf::from("scn2.mat")
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn final_product_scope_excludes_grid_and_smoothing_intermediates() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rustamps-final-products-{stamp}"));
        fs::create_dir_all(root.join("PATCH_1")).unwrap();
        for name in ["phuw2.mat", "uw_grid.mat", "scla2.mat", "scla_smooth2.mat"] {
            fs::write(root.join(name), []).unwrap();
        }

        assert_eq!(
            artifact_paths(&root, None, true).unwrap(),
            vec![PathBuf::from("phuw2.mat"), PathBuf::from("scla2.mat")]
        );
        fs::remove_dir_all(root).unwrap();
    }
}
