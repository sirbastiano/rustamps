mod artifacts;
mod compare;
mod policy;
#[cfg(test)]
mod profile_tests;
mod report;
mod value_compare;
mod value_numeric;
mod value_result;
mod value_support;

pub use compare::{verify_paths, verify_paths_through_stage, verify_paths_with_scope, VerifyError};
pub use report::{FileComparison, OutlierSummary, VerificationReport};
