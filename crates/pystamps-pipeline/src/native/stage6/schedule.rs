const AUTO_MAX_IFG_WORKERS: usize = 4;
const THREADS_PER_ACTIVE_SOLVE: usize = 3;
const TOTAL_ACTIVE_CELL_BUDGET: usize = 9_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Schedule {
    pub rayon_workers: usize,
    pub effective_ifg_workers: usize,
}

pub fn choose(requested: usize, rows: usize, cols: usize, count: usize) -> Schedule {
    choose_for_rayon(requested, rayon::current_num_threads(), rows, cols, count)
}

fn choose_for_rayon(
    requested: usize,
    rayon_workers: usize,
    rows: usize,
    cols: usize,
    count: usize,
) -> Schedule {
    let rayon_workers = rayon_workers.max(1);
    if count == 0 {
        return Schedule {
            rayon_workers,
            effective_ifg_workers: 0,
        };
    }
    let requested_limit = if requested == 0 {
        AUTO_MAX_IFG_WORKERS
    } else {
        requested.min(AUTO_MAX_IFG_WORKERS)
    };
    let worker_limit = (rayon_workers / THREADS_PER_ACTIVE_SOLVE).max(1);
    let cells = rows.checked_mul(cols).unwrap_or(usize::MAX).max(1);
    let memory_limit = (TOTAL_ACTIVE_CELL_BUDGET / cells).max(1);
    let effective_ifg_workers = requested_limit
        .min(worker_limit)
        .min(memory_limit)
        .min(count);
    Schedule {
        rayon_workers,
        effective_ifg_workers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_reaches_four_only_when_threads_and_cells_allow_it() {
        assert_eq!(
            choose_for_rayon(0, 12, 932, 2_361, 75).effective_ifg_workers,
            4
        );
        assert_eq!(
            choose_for_rayon(0, 6, 932, 2_361, 75).effective_ifg_workers,
            2
        );
        assert_eq!(
            choose_for_rayon(0, 12, 1_773, 4_378, 74).effective_ifg_workers,
            1
        );
    }

    #[test]
    fn explicit_sizes_are_safe_upper_bounds() {
        assert_eq!(
            choose_for_rayon(1, 12, 100, 100, 20).effective_ifg_workers,
            1
        );
        assert_eq!(
            choose_for_rayon(2, 12, 100, 100, 20).effective_ifg_workers,
            2
        );
        assert_eq!(
            choose_for_rayon(4, 12, 100, 100, 20).effective_ifg_workers,
            4
        );
        assert_eq!(
            choose_for_rayon(4, 6, 100, 100, 20).effective_ifg_workers,
            2
        );
        assert_eq!(
            choose_for_rayon(4, 12, 100, 100, 1).effective_ifg_workers,
            1
        );
        assert_eq!(
            choose_for_rayon(4, 12, 100, 100, 0).effective_ifg_workers,
            0
        );
        assert_eq!(
            choose_for_rayon(4, 12, usize::MAX, 2, 20).effective_ifg_workers,
            1
        );
        assert_eq!(
            choose_for_rayon(4, 12, 1_773, 4_378, 74).effective_ifg_workers,
            1
        );
    }
}
