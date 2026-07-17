use std::fs;
use std::io;
use std::path::Path;

const PATCH_PRODUCTS: &[(u8, &[&str])] = &[
    (2, &["pm1.mat"]),
    (3, &["select1.mat"]),
    (4, &["weed1.mat"]),
    (
        5,
        &[
            "ps2.mat", "ph2.mat", "pm2.mat", "bp2.mat", "rc2.mat", "da2.mat", "hgt2.mat",
            "la2.mat", "inc2.mat",
        ],
    ),
];

const ROOT_PRODUCTS: &[(u8, &[&str])] = &[
    (
        5,
        &[
            "ps2.mat",
            "ph2.mat",
            "pm2.mat",
            "bp2.mat",
            "rc2.mat",
            "da2.mat",
            "hgt2.mat",
            "la2.mat",
            "inc2.mat",
            "ifgstd2.mat",
            "psver.mat",
        ],
    ),
    (
        6,
        &[
            "uw_grid.mat",
            "uw_interp.mat",
            "uw_space_time.mat",
            "uw_phaseuw.mat",
            "uw_stat_cost.mat",
            "phuw2.mat",
            // Stage 6 consumes the previous Stage 7 smooth estimate as
            // feedback. Preserve it on a Stage 6 rerun, but invalidate it
            // whenever Stage 5 or an earlier stage changes.
            "scla_smooth2.mat",
        ],
    ),
    (7, &["scla2.mat"]),
    (8, &["scn2.mat"]),
];

pub(crate) fn invalidate_downstream(
    stage: u8,
    scope: &str,
    target: &Path,
    root: &Path,
) -> io::Result<()> {
    if scope == "patch" {
        remove_products(target, PATCH_PRODUCTS, |product_stage| {
            product_stage > stage
        })?;
        if stage <= 5 {
            remove_products(root, ROOT_PRODUCTS, |_| true)?;
        }
    } else {
        remove_products(root, ROOT_PRODUCTS, |product_stage| product_stage > stage)?;
    }
    Ok(())
}

fn remove_products(
    directory: &Path,
    products: &[(u8, &[&str])],
    should_remove: impl Fn(u8) -> bool,
) -> io::Result<()> {
    for &(stage, names) in products {
        if !should_remove(stage) {
            continue;
        }
        for &name in names {
            let path = directory.join(name);
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (std::path::PathBuf, std::path::PathBuf) {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "pystamps-invalidation-{}-{stamp}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        let patch = root.join("PATCH_1");
        fs::create_dir_all(&patch).unwrap();
        for &(_, names) in PATCH_PRODUCTS {
            for &name in names {
                fs::write(patch.join(name), []).unwrap();
            }
        }
        for &(_, names) in ROOT_PRODUCTS {
            for &name in names {
                fs::write(root.join(name), []).unwrap();
            }
        }
        (root, patch)
    }

    #[test]
    fn patch_rerun_removes_only_later_patch_products_and_all_merged_products() {
        let (root, patch) = fixture();
        invalidate_downstream(2, "patch", &patch, &root).unwrap();
        assert!(patch.join("pm1.mat").exists());
        assert!(!patch.join("select1.mat").exists());
        assert!(!patch.join("weed1.mat").exists());
        assert!(!patch.join("ph2.mat").exists());
        assert!(!root.join("ps2.mat").exists());
        assert!(!root.join("phuw2.mat").exists());
        assert!(!root.join("scla2.mat").exists());
        assert!(!root.join("scn2.mat").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn merged_rerun_preserves_its_stage_and_removes_later_products() {
        let (root, patch) = fixture();
        invalidate_downstream(6, "merged", &root, &root).unwrap();
        assert!(root.join("phuw2.mat").exists());
        assert!(root.join("scla_smooth2.mat").exists());
        assert!(!root.join("scla2.mat").exists());
        assert!(!root.join("scn2.mat").exists());
        fs::remove_dir_all(root).unwrap();
        let _ = patch;
    }
}
