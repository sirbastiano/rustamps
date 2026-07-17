use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TransactionError {
    #[error("transaction I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("transaction source is outside its staging directory: {0}")]
    OutsideStaging(PathBuf),
    #[error("commit marker {0} must be included in the bundle")]
    MissingMarker(String),
}

pub struct StageTransaction {
    target: PathBuf,
    staging: PathBuf,
    committed: bool,
}

impl StageTransaction {
    pub fn begin(target: impl AsRef<Path>, label: &str) -> Result<Self, TransactionError> {
        let target = target.as_ref().to_path_buf();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let staging = target
            .join(".pystamps-tmp")
            .join(format!("{label}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&staging)?;
        Ok(Self {
            target,
            staging,
            committed: false,
        })
    }

    pub fn path(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.staging.join(relative)
    }

    pub fn commit(self, files: &[&str], marker: &str) -> Result<(), TransactionError> {
        self.commit_with_removals(files, marker, &[])
    }

    pub fn commit_with_removals(
        mut self,
        files: &[&str],
        marker: &str,
        removals: &[&str],
    ) -> Result<(), TransactionError> {
        if !files.contains(&marker) {
            return Err(TransactionError::MissingMarker(marker.to_owned()));
        }
        let ordered = std::iter::once(marker)
            .chain(files.iter().copied().filter(|name| *name != marker))
            .chain(
                removals
                    .iter()
                    .copied()
                    .filter(|name| *name != marker && !files.contains(name)),
            )
            .collect::<Vec<_>>();
        for name in files {
            let source = self.staging.join(name);
            if !safe_relative(name) || !source.exists() {
                return Err(TransactionError::OutsideStaging(source));
            }
        }
        if removals.iter().any(|name| !safe_relative(name)) {
            return Err(TransactionError::OutsideStaging(
                self.staging.join("invalid-removal"),
            ));
        }
        let backup_root = self.staging.join(".backup");
        let mut backed_up = Vec::new();
        let mut published = Vec::new();
        // Once destinations move, preserve staging on an unrecoverable error.
        self.committed = true;
        for name in &ordered {
            let destination = self.target.join(name);
            if destination.exists() {
                let backup = backup_root.join(name);
                let result = backup
                    .parent()
                    .map_or(Ok(()), fs::create_dir_all)
                    .and_then(|_| fs::rename(&destination, &backup));
                if let Err(error) = result {
                    return self.rollback(error, &published, &backed_up, marker);
                }
                backed_up.push(*name);
            }
        }
        let publish_order = files
            .iter()
            .copied()
            .filter(|name| *name != marker)
            .chain(std::iter::once(marker));
        for name in publish_order {
            let source = self.staging.join(name);
            let destination = self.target.join(name);
            let result = destination
                .parent()
                .map_or(Ok(()), fs::create_dir_all)
                .and_then(|_| fs::rename(&source, &destination));
            if let Err(error) = result {
                return self.rollback(error, &published, &backed_up, marker);
            }
            published.push(name);
        }
        fs::remove_dir_all(&self.staging)?;
        Ok(())
    }

    fn rollback(
        &self,
        cause: std::io::Error,
        published: &[&str],
        backed_up: &[&str],
        marker: &str,
    ) -> Result<(), TransactionError> {
        for name in published.iter().rev() {
            remove_path(&self.target.join(name))?;
        }
        for name in backed_up
            .iter()
            .copied()
            .filter(|name| *name != marker)
            .chain(backed_up.iter().copied().filter(|name| *name == marker))
        {
            let backup = self.staging.join(".backup").join(name);
            let destination = self.target.join(name);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::rename(backup, destination)?;
        }
        fs::remove_dir_all(&self.staging)?;
        Err(TransactionError::Io(cause))
    }
}

fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path).is_relative()
        && Path::new(path)
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
}

fn remove_path(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path)
    } else if path.exists() {
        fs::remove_file(path)
    } else {
        Ok(())
    }
}

impl Drop for StageTransaction {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir_all(&self.staging);
        }
    }
}

pub fn atomic_write(path: impl AsRef<Path>, bytes: &[u8]) -> Result<(), TransactionError> {
    let path = path.as_ref();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let temp = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let backup = parent.join(format!(".{name}.bak-{}-{nonce}", std::process::id()));
    let mut file = File::create(&temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    let had_destination = path.exists();
    if had_destination {
        if let Err(error) = fs::rename(path, &backup) {
            let _ = fs::remove_file(&temp);
            return Err(error.into());
        }
    }
    if let Err(error) = fs::rename(&temp, path) {
        if had_destination {
            let _ = fs::rename(&backup, path);
        }
        let _ = fs::remove_file(&temp);
        return Err(error.into());
    }
    if had_destination {
        fs::remove_file(backup)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_bundle_publish_restores_outputs_and_completion_marker() {
        let root = std::env::temp_dir().join(format!(
            "pystamps-transaction-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("good"), b"old-good").unwrap();
        fs::write(root.join("marker"), b"old-marker").unwrap();
        fs::write(root.join("blocked"), b"parent-is-a-file").unwrap();
        let transaction = StageTransaction::begin(&root, "rollback").unwrap();
        fs::write(transaction.path("good"), b"new-good").unwrap();
        fs::create_dir_all(transaction.path("blocked")).unwrap();
        fs::write(transaction.path("blocked/file"), b"new-blocked").unwrap();
        fs::write(transaction.path("marker"), b"new-marker").unwrap();

        assert!(transaction
            .commit(&["good", "blocked/file", "marker"], "marker")
            .is_err());
        assert_eq!(fs::read(root.join("good")).unwrap(), b"old-good");
        assert_eq!(fs::read(root.join("marker")).unwrap(), b"old-marker");
        assert_eq!(fs::read(root.join("blocked")).unwrap(), b"parent-is-a-file");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn successful_bundle_can_remove_stale_optional_output() {
        let root = std::env::temp_dir().join(format!(
            "pystamps-removal-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("marker"), b"old").unwrap();
        fs::write(root.join("optional"), b"stale").unwrap();
        let transaction = StageTransaction::begin(&root, "removal").unwrap();
        fs::write(transaction.path("marker"), b"new").unwrap();
        transaction
            .commit_with_removals(&["marker"], "marker", &["optional"])
            .unwrap();
        assert_eq!(fs::read(root.join("marker")).unwrap(), b"new");
        assert!(!root.join("optional").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_write_replaces_an_existing_file_without_leaking_backups() {
        let root = std::env::temp_dir().join(format!(
            "pystamps-atomic-write-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = root.join("metadata.txt");
        atomic_write(&path, b"first").unwrap();
        atomic_write(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
