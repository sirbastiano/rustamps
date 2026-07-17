use std::fs;
use std::path::Path;

pub fn reject_unsupported_spatial_inputs(patch: &Path) -> Result<(), String> {
    let mut found = Vec::new();
    for root in [Some(patch), patch.parent()].into_iter().flatten() {
        for name in [
            "look_angle.1.in",
            "heading.1.in",
            "lambda.1.in",
            "slc_osfactor.1.in",
        ] {
            let path = root.join(name);
            if path.is_file() {
                found.push(path);
            }
        }
        for entry in fs::read_dir(root).map_err(|error| error.to_string())? {
            let path = entry.map_err(|error| error.to_string())?.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if path.is_file() && name.starts_with("bperp_") && name.ends_with(".1.in") {
                found.push(path);
            }
        }
    }
    if found.is_empty() {
        return Ok(());
    }
    found.sort();
    found.dedup();
    let names = found
        .iter()
        .filter_map(|path| path.file_name()?.to_str())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "Stage 1 cannot safely consume legacy geometry/wavelength/oversampling inputs ({names}); native legacy metadata ingestion is not implemented"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pystamps_io::{write_mat, MatArray, MatFile, MatValue};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir(PathBuf);

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    impl TempDir {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "pystamps-stage1-legacy-{}-{nonce}-{}",
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(path.join("PATCH_1")).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn spatial_legacy_inputs_fail_closed() {
        let temp = TempDir::new();
        fs::write(temp.0.join("look_angle.1.in"), []).unwrap();
        fs::write(temp.0.join("heading.1.in"), []).unwrap();
        fs::write(temp.0.join("lambda.1.in"), []).unwrap();
        fs::write(temp.0.join("slc_osfactor.1.in"), []).unwrap();
        fs::write(temp.0.join("PATCH_1/bperp_20200113.1.in"), []).unwrap();

        let error = reject_unsupported_spatial_inputs(&temp.0.join("PATCH_1")).unwrap_err();

        assert!(error.contains("native legacy metadata ingestion is not implemented"));
        assert!(error.contains("look_angle.1.in"));
        assert!(error.contains("heading.1.in"));
        assert!(error.contains("lambda.1.in"));
        assert!(error.contains("slc_osfactor.1.in"));
        assert!(error.contains("bperp_20200113.1.in"));
    }

    #[test]
    fn snap_synthesis_bypasses_legacy_input_guard() {
        let temp = TempDir::new();
        fs::create_dir_all(temp.0.join("diff0")).unwrap();
        fs::create_dir_all(temp.0.join("rslc")).unwrap();
        fs::write(temp.0.join("look_angle.1.in"), []).unwrap();
        fs::write(
            temp.0.join("diff0/20200101_20200113.base"),
            "initial_baseline(TCN): 0 1 2\ninitial_baseline_rate: 0 0 0\n",
        )
        .unwrap();
        fs::write(
            temp.0.join("rslc/20200101.rslc.par"),
            "range_pixel_spacing: 10\nnear_range_slc: 800000\ncenter_range_slc: 800000\nsar_to_earth_center: 7071000\nearth_radius_below_sensor: 6371000\nazimuth_lines: 100\nprf: 1000\nheading: 190\nradar_frequency: 5.4e9\n",
        )
        .unwrap();

        let metadata =
            super::super::metadata::resolve(&temp.0.join("PATCH_1"), &[1.0, 0.0, 0.0], 1).unwrap();

        assert_eq!(metadata.master, 20_200_101);
        assert_eq!(metadata.days, [20_200_113]);
        assert!(metadata.bperp_mat.is_some());
        assert!(metadata.wavelength.is_some());
        assert!(super::super::validate_sensor_params(&temp.0.join("PATCH_1"), &metadata).is_err());
        let mut params = MatFile::new();
        for (key, value) in [("heading", 190.0), ("lambda", 299_792_458.0 / 5.4e9)] {
            params.insert(
                key.to_owned(),
                MatValue::F64(MatArray {
                    shape: vec![1, 1],
                    values: vec![value],
                }),
            );
        }
        write_mat(temp.0.join("parms.mat"), &params).unwrap();
        super::super::validate_sensor_params(&temp.0.join("PATCH_1"), &metadata).unwrap();
    }
}
