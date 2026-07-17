use std::path::PathBuf;

use pystamps_io::{read_mat, MatArray, MatValue};

fn dataset_patch() -> PathBuf {
    std::env::var_os("PYSTAMPS_REAL_PATCH").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("inputs_and_outputs/InSAR_dataset_test/PATCH_1")
        },
        PathBuf::from,
    )
}

fn assert_shape<T>(array: &MatArray<T>) {
    assert_eq!(array.shape.iter().product::<usize>(), array.values.len());
}

#[test]
#[ignore = "opt-in: reads the large local InSAR_dataset_test MAT files"]
fn reads_bundled_real_v73_ps_and_phase_products() {
    let patch = dataset_patch();
    assert!(
        patch.is_dir(),
        "missing real-data patch {}",
        patch.display()
    );

    let ps = read_mat(patch.join("ps1.mat")).unwrap();
    assert!(ps.len() >= 10);
    match ps.get("xy").expect("ps1.mat:xy") {
        MatValue::F32(value) => {
            assert_shape(value);
            assert_eq!(value.shape.get(1), Some(&3));
        }
        other => panic!("ps1.mat:xy has unexpected type {other:?}"),
    }

    let phase = read_mat(patch.join("ph1.mat")).unwrap();
    match phase.get("ph").expect("ph1.mat:ph") {
        MatValue::ComplexF32(value) => {
            assert_shape(value);
            assert_eq!(value.shape.len(), 2);
            assert!(value.shape.iter().all(|&size| size > 1));
        }
        other => panic!("ph1.mat:ph has unexpected type {other:?}"),
    }
}

#[test]
#[ignore = "opt-in: reads the local generated Level-5 stage artifact when present"]
fn reads_real_level5_stage_product_when_available() {
    let path = dataset_patch().join("select1.mat");
    assert!(
        path.is_file(),
        "missing real-data artifact {}",
        path.display()
    );
    let payload = read_mat(path).unwrap();
    assert!(!payload.is_empty());
    for value in payload.values() {
        if let Some(shape) = value.shape() {
            assert!(shape.iter().product::<usize>() > 0);
        }
    }
}
