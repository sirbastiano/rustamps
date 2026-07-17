use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{read_mat, write_mat, MatArray, MatFile, MatValue};

use super::{discover, write_sensor_params};

#[test]
fn snap_sensor_metadata_is_derived_and_persisted() {
    let root = std::env::temp_dir().join(format!(
        "rustamps-snap-sensor-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    for directory in ["rslc", "diff0", "geo"] {
        fs::create_dir_all(root.join(directory)).unwrap();
    }
    fs::write(root.join("width.txt"), "1\n").unwrap();
    fs::write(root.join("len.txt"), "1\n").unwrap();
    fs::write(root.join("rslc/20200101.rslc"), []).unwrap();
    fs::write(
        root.join("rslc/20200101.rslc.par"),
        "heading: 190.5 degrees\nradar_frequency: 5.4e9 Hz\n",
    )
    .unwrap();
    fs::write(root.join("diff0/20200101_20200113.diff"), []).unwrap();
    for name in [
        "geo/20200101.lon",
        "geo/20200101.lat",
        "geo/elevation_dem.rdc",
    ] {
        fs::write(root.join(name), []).unwrap();
    }

    let discovered = discover::discover(&root, Some("20200101")).unwrap();
    assert_eq!(discovered.heading, 190.5);
    assert!((discovered.wavelength - 299_792_458.0 / 5.4e9).abs() < 1.0e-15);
    let mut stale = MatFile::new();
    for key in ["heading", "lambda"] {
        stale.insert(
            key.to_owned(),
            MatValue::F64(MatArray {
                shape: vec![1, 1],
                values: vec![1.0],
            }),
        );
    }
    write_mat(root.join("parms.mat"), &stale).unwrap();
    let output = root.join("prepared-parms.mat");
    write_sensor_params(&root, &output, discovered.heading, discovered.wavelength).unwrap();
    let params = read_mat(output).unwrap();
    let scalar = |key| match params.get(key) {
        Some(MatValue::F64(value)) => value.values[0],
        _ => panic!("missing f64 {key}"),
    };
    assert_eq!(scalar("heading"), discovered.heading);
    assert_eq!(scalar("lambda"), discovered.wavelength);
    let _ = fs::remove_dir_all(root);
}
