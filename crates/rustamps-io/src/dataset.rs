use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

const PATCH_PREFIX: &str = "PATCH_";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetLayout {
    pub root: PathBuf,
    pub patches: Vec<PathBuf>,
    pub patch_list_file: Option<PathBuf>,
}

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("dataset root does not exist: {0}")]
    MissingRoot(PathBuf),
    #[error("failed to read dataset layout: {0}")]
    Io(#[from] std::io::Error),
    #[error("patch.list references missing patch directories: {0}")]
    MissingPatches(String),
    #[error("patch.list contains unsafe patch names: {0}")]
    UnsafePatchNames(String),
}

fn patch_sort_key(path: &Path) -> (u64, String) {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let suffix = name.strip_prefix(PATCH_PREFIX).unwrap_or(name);
    (suffix.parse().unwrap_or(u64::MAX), name.to_owned())
}

pub fn discover_dataset(root: impl AsRef<Path>) -> Result<DatasetLayout, DatasetError> {
    let root = root.as_ref();
    if !root.exists() {
        return Err(DatasetError::MissingRoot(root.to_path_buf()));
    }
    let root = fs::canonicalize(root)?;
    let patch_list = root.join("patch.list");
    let (patches, patch_list_file) = if patch_list.exists() {
        let text = fs::read_to_string(&patch_list)?;
        let names: Vec<_> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        let unsafe_names = names
            .iter()
            .filter(|name| {
                let path = Path::new(name);
                path.is_absolute() || path.components().count() != 1
            })
            .copied()
            .collect::<Vec<_>>();
        if !unsafe_names.is_empty() {
            return Err(DatasetError::UnsafePatchNames(unsafe_names.join(", ")));
        }
        let patches: Vec<_> = names.iter().map(|name| root.join(name)).collect();
        let missing: Vec<_> = names
            .iter()
            .zip(&patches)
            .filter_map(|(name, path)| (!path.is_dir()).then_some(*name))
            .collect();
        if !missing.is_empty() {
            return Err(DatasetError::MissingPatches(missing.join(", ")));
        }
        let mut resolved = Vec::with_capacity(patches.len());
        let mut escaped = Vec::new();
        for (name, patch) in names.iter().zip(patches) {
            let canonical = fs::canonicalize(&patch)?;
            if canonical == patch {
                resolved.push(canonical);
            } else {
                escaped.push(*name);
            }
        }
        if !escaped.is_empty() {
            return Err(DatasetError::UnsafePatchNames(escaped.join(", ")));
        }
        (resolved, Some(patch_list))
    } else {
        let mut patches = Vec::new();
        let mut escaped = Vec::new();
        for entry in fs::read_dir(&root)? {
            let path = entry?.path();
            let is_patch = path.is_dir()
                && path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|name| name.starts_with(PATCH_PREFIX));
            if is_patch {
                let canonical = fs::canonicalize(&path)?;
                if canonical == path {
                    patches.push(canonical);
                } else {
                    escaped.push(
                        path.file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
        }
        if !escaped.is_empty() {
            return Err(DatasetError::UnsafePatchNames(escaped.join(", ")));
        }
        patches.sort_by_key(|path| patch_sort_key(path));
        (patches, None)
    };
    Ok(DatasetLayout {
        root,
        patches,
        patch_list_file,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_sort_is_numeric() {
        let mut paths = vec![PathBuf::from("PATCH_10"), PathBuf::from("PATCH_2")];
        paths.sort_by_key(|path| patch_sort_key(path));
        assert_eq!(paths[0], PathBuf::from("PATCH_2"));
    }

    #[test]
    fn patch_list_cannot_escape_the_dataset_root() {
        let root = std::env::temp_dir().join(format!(
            "rustamps-layout-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("patch.list"), "../outside\n/absolute\n").unwrap();

        let error = discover_dataset(&root).unwrap_err();

        assert!(matches!(error, DatasetError::UnsafePatchNames(_)));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn patch_directory_links_are_rejected() {
        use std::os::unix::fs::symlink;

        let base = std::env::temp_dir().join(format!(
            "rustamps-layout-link-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let root = base.join("dataset");
        let outside = base.join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("PATCH_1")).unwrap();
        fs::write(root.join("patch.list"), "PATCH_1\n").unwrap();

        assert!(matches!(
            discover_dataset(&root).unwrap_err(),
            DatasetError::UnsafePatchNames(_)
        ));
        fs::remove_file(root.join("patch.list")).unwrap();
        assert!(matches!(
            discover_dataset(&root).unwrap_err(),
            DatasetError::UnsafePatchNames(_)
        ));
        fs::remove_file(root.join("PATCH_1")).unwrap();
        fs::create_dir(root.join("PATCH_2")).unwrap();
        symlink(root.join("PATCH_2"), root.join("PATCH_1")).unwrap();
        fs::write(root.join("patch.list"), "PATCH_1\n").unwrap();
        assert!(matches!(
            discover_dataset(&root).unwrap_err(),
            DatasetError::UnsafePatchNames(_)
        ));
        fs::remove_file(root.join("PATCH_1")).unwrap();
        symlink(&root, root.join("PATCH_1")).unwrap();
        assert!(matches!(
            discover_dataset(&root).unwrap_err(),
            DatasetError::UnsafePatchNames(_)
        ));
        fs::remove_file(root.join("PATCH_1")).unwrap();
        fs::remove_dir_all(base).unwrap();
    }
}
