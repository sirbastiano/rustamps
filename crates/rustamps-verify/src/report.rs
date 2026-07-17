use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutlierSummary {
    pub key: String,
    pub count: usize,
    pub total: usize,
    pub fraction: f64,
    pub allowed_fraction: f64,
    pub max_abs: f64,
    pub max_abs_limit: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileComparison {
    pub path: String,
    pub ok: bool,
    pub message: String,
    pub failing_key: Option<String>,
    pub max_abs: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outliers: Vec<OutlierSummary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VerificationReport {
    pub comparisons: Vec<FileComparison>,
}

impl VerificationReport {
    pub fn ok(&self) -> bool {
        self.comparisons.iter().all(|comparison| comparison.ok)
    }
}
