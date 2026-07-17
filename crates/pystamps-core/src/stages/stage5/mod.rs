mod merge;
mod promote;
mod science;
mod types;

pub use merge::{merge_patches, merge_patches_with_heading};
pub use promote::promote_patch;
pub use science::*;
pub use types::{
    NoOverlapBounds, PromotedPatch, Stage5Error, Stage5Merged, Stage5PatchInput, Stage5Row,
};
