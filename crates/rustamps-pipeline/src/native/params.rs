use std::path::{Path, PathBuf};

use rustamps_io::{read_mat, MatFile, MatValue};

use super::mat::numeric_f64;

pub struct Params {
    values: MatFile,
}

impl Params {
    pub fn load(root: &Path) -> Result<Self, String> {
        let mut values = if let Some(path) = find(root, "parms.mat") {
            read_mat(path).map_err(|error| error.to_string())?
        } else {
            MatFile::new()
        };
        let local = root.join("localparms.mat");
        if local.is_file() {
            values.extend(read_mat(local).map_err(|error| error.to_string())?);
        }
        Ok(Self { values })
    }

    pub fn scalar(&self, key: &str, default: f64) -> Result<f64, String> {
        if !self.values.contains_key(key) {
            return Ok(default);
        }
        numeric_f64(&self.values, key)?
            .first()
            .copied()
            .ok_or_else(|| format!("parameter {key} is empty"))
    }

    pub fn text(&self, key: &str, default: &str) -> Result<String, String> {
        let Some(value) = self.values.get(key) else {
            return Ok(default.to_owned());
        };
        let text = match value {
            MatValue::Char(value) => String::from_utf16_lossy(&value.values),
            MatValue::U8(value) => String::from_utf8_lossy(&value.values).into_owned(),
            MatValue::I8(value) => {
                let bytes = value
                    .values
                    .iter()
                    .map(|&item| item as u8)
                    .collect::<Vec<_>>();
                String::from_utf8_lossy(&bytes).into_owned()
            }
            _ => return Err(format!("parameter {key} is not text")),
        };
        Ok(text.trim_matches(char::from(0)).trim().to_owned())
    }

    pub fn flag(&self, key: &str, default: bool) -> Result<bool, String> {
        let fallback = if default { "y" } else { "n" };
        match self.text(key, fallback)?.to_ascii_lowercase().as_str() {
            "y" | "yes" | "true" | "1" => Ok(true),
            "n" | "no" | "false" | "0" | "" => Ok(false),
            value => Err(format!("parameter {key} has invalid flag {value}")),
        }
    }

    pub fn indices(&self, key: &str) -> Result<Vec<usize>, String> {
        if !self.values.contains_key(key) {
            return Ok(Vec::new());
        }
        numeric_f64(&self.values, key)?
            .into_iter()
            .filter_map(|value| {
                if value.is_nan() || value == 0.0 {
                    None
                } else if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                    Some(Err(format!(
                        "parameter {key} contains invalid one-based index {value}"
                    )))
                } else {
                    Some(Ok(value as usize - 1))
                }
            })
            .collect()
    }

    pub fn vector(&self, key: &str, default: &[f64]) -> Result<Vec<f64>, String> {
        if self.values.contains_key(key) {
            numeric_f64(&self.values, key)
        } else {
            Ok(default.to_vec())
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }
}

fn find(root: &Path, name: &str) -> Option<PathBuf> {
    [root.to_path_buf(), root.parent()?.to_path_buf()]
        .into_iter()
        .map(|candidate| candidate.join(name))
        .find(|path| path.exists())
}

#[cfg(test)]
mod tests {
    use rustamps_io::{MatArray, MatValue};

    use super::*;

    #[test]
    fn zero_and_nan_drop_sentinels_are_ignored() {
        let mut values = MatFile::new();
        values.insert(
            "drop_ifg_index".to_owned(),
            MatValue::F64(MatArray {
                shape: vec![1, 4],
                values: vec![0.0, f64::NAN, 2.0, 4.0],
            }),
        );
        assert_eq!(Params { values }.indices("drop_ifg_index").unwrap(), [1, 3]);
    }
}
