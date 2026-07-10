#[path = "stage8_native_lstsq.rs"]
mod stage8_native_lstsq;
#[path = "stage8_native_noise.rs"]
mod stage8_native_noise;

pub use self::stage8_native_lstsq::{stage8_weighted_lstsq_diagonal, stage8_weighted_lstsq_full};
pub use self::stage8_native_noise::stage8_edge_noise;
