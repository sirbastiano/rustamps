use crate::OutlierSummary;

#[derive(Debug, PartialEq)]
pub(crate) struct CompareFailure {
    pub key: String,
    pub message: String,
    pub max_abs: Option<f64>,
    pub outliers: Vec<OutlierSummary>,
}

pub(crate) type CompareResult = Result<Vec<OutlierSummary>, CompareFailure>;

pub(crate) fn failure(key: &str, message: &str, max_abs: Option<f64>) -> CompareFailure {
    CompareFailure {
        key: key.to_owned(),
        message: message.to_owned(),
        max_abs,
        outliers: Vec::new(),
    }
}

pub(crate) fn numeric_failure(
    key: &str,
    message: String,
    summary: OutlierSummary,
) -> CompareFailure {
    CompareFailure {
        key: key.to_owned(),
        message,
        max_abs: Some(summary.max_abs),
        outliers: vec![summary],
    }
}
