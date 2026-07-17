use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rustamps_io::{prepare_snap, read_mat, MatValue, SnapPrepOptions};

#[test]
fn snap_prep_reads_big_endian_rasters_and_force_replaces_atomically() {
    let root = fixture();
    write_fixture(&root);
    let options = |force| SnapPrepOptions {
        master_date: Some("20200101"),
        amp_dispersion: 0.4,
        range_patches: 1,
        azimuth_patches: 1,
        range_overlap: 0,
        azimuth_overlap: 0,
        force,
    };

    let summary = prepare_snap(&root, options(false)).unwrap();
    assert_eq!(summary.patch_count, 1);
    assert_eq!(summary.candidate_count, 6);
    let params = read_mat(root.join("parms.mat")).unwrap();
    for key in ["heading", "lambda"] {
        assert!(matches!(params.get(key), Some(MatValue::F64(_))));
    }
    let patch = root.join("PATCH_1");
    assert_eq!(
        fs::read_to_string(patch.join("pscands.1.ij")).unwrap(),
        candidate_rows()
    );
    assert_eq!(fs::metadata(patch.join("pscands.1.ph")).unwrap().len(), 48);
    let amplitudes = fs::read(patch.join("mean_amp.flt")).unwrap();
    assert!(amplitudes
        .chunks_exact(4)
        .all(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()) == 2.0));

    fs::write(patch.join("obsolete"), b"old").unwrap();
    assert!(prepare_snap(&root, options(false)).is_err());
    assert!(patch.join("obsolete").is_file());
    prepare_snap(&root, options(true)).unwrap();
    assert!(!patch.join("obsolete").exists());
    assert_eq!(
        fs::read_to_string(root.join("patch.list")).unwrap(),
        "PATCH_1\n"
    );
    fs::remove_dir_all(root).unwrap();
}

fn write_fixture(root: &Path) {
    for directory in ["rslc", "diff0", "geo"] {
        fs::create_dir_all(root.join(directory)).unwrap();
    }
    fs::write(root.join("width.txt"), "3\n").unwrap();
    fs::write(root.join("len.txt"), "2\n").unwrap();
    write_complex(root.join("rslc/20200101.rslc"), &[1.0; 6]);
    fs::write(
        root.join("rslc/20200101.rslc.par"),
        "heading: 190.5 degrees\nradar_frequency: 5.4e9 Hz\n",
    )
    .unwrap();
    write_complex(root.join("rslc/20200113.rslc"), &[2.0; 6]);
    write_complex(
        root.join("diff0/20200101_20200113.diff"),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    write_real(
        root.join("geo/20200101.lon"),
        &[10.0, 11.0, 12.0, 10.0, 11.0, 12.0],
    );
    write_real(
        root.join("geo/20200101.lat"),
        &[40.0, 40.0, 40.0, 41.0, 41.0, 41.0],
    );
    write_real(root.join("geo/elevation_dem.rdc"), &[100.0; 6]);
}

fn write_complex(path: PathBuf, real: &[f32]) {
    let bytes = real
        .iter()
        .flat_map(|value| value.to_be_bytes().into_iter().chain(0.0_f32.to_be_bytes()))
        .collect::<Vec<_>>();
    fs::write(path, bytes).unwrap();
}

fn write_real(path: PathBuf, values: &[f32]) {
    fs::write(
        path,
        values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}

fn candidate_rows() -> String {
    "1 0 0\n2 0 1\n3 0 2\n4 1 0\n5 1 1\n6 1 2\n".to_owned()
}

fn fixture() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("rustamps-snap-prep-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    root
}
