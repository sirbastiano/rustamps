use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageResult {
    pub stage: u8,
    pub scope: String,
    pub target: String,
    pub status: String,
    pub details: String,
    pub duration_sec: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PipelineReport {
    pub results: Vec<StageResult>,
}

impl PipelineReport {
    pub fn failures(&self) -> impl Iterator<Item = &StageResult> {
        self.results
            .iter()
            .filter(|result| result.status == "failed")
    }

    pub fn ok(&self) -> bool {
        self.failures().next().is_none()
    }
}
