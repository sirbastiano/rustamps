#[path = "stage4_edge_stats_api.rs"]
mod stage4_edge_stats_api;
#[path = "stage4_edge_stats_core.rs"]
mod stage4_edge_stats_core;
#[path = "stage4_edge_stats_stats.rs"]
mod stage4_edge_stats_stats;

pub use self::stage4_edge_stats_api::stage4_edge_stats;
