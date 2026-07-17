use super::{ConfigError, RunConfig};

pub(super) fn reject_inert_options(config: &RunConfig) -> Result<(), ConfigError> {
    let runtime = &config.runtime;
    let inert = [
        (runtime.io_workers != 8, "runtime.io_workers"),
        (
            runtime.stage2_native_threads != 0,
            "runtime.stage2_native_threads",
        ),
        (
            runtime.stage7_chunk_ps != 100_000,
            "runtime.stage7_chunk_ps",
        ),
        (
            runtime.stage8_chunk_edges != 200_000,
            "runtime.stage8_chunk_edges",
        ),
        (
            !runtime.enable_mat_stage_cache,
            "runtime.enable_mat_stage_cache",
        ),
        (
            runtime.stage2_checkpoint_mode != "final",
            "runtime.stage2_checkpoint_mode",
        ),
        (
            runtime.stage2_checkpoint_interval != 1,
            "runtime.stage2_checkpoint_interval",
        ),
        (runtime.stage2_debug, "runtime.stage2_debug"),
        (runtime.stage4_debug, "runtime.stage4_debug"),
    ];
    if let Some((_, field)) = inert.into_iter().find(|(changed, _)| *changed) {
        return Err(ConfigError::Unsupported(format!(
            "{field} has no native implementation; omit it"
        )));
    }
    if !runtime.kernel_backend_overrides.is_empty()
        || !runtime.stage2_patch_backend_overrides.is_empty()
    {
        return Err(ConfigError::Unsupported(
            "per-kernel backend overrides were removed; all kernels are native".to_owned(),
        ));
    }
    if config.compatibility.reference_root.is_some() || config.compatibility.strict_reference {
        return Err(ConfigError::Unsupported(
            "legacy reference replay compatibility was removed; use `pystamps verify`".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_noop_runtime_and_reference_replay_options() {
        let mut config = RunConfig::default();
        config.runtime.io_workers = 2;
        assert!(reject_inert_options(&config).is_err());

        let mut config = RunConfig::default();
        config
            .runtime
            .kernel_backend_overrides
            .insert("stage7_scla".to_owned(), "native".to_owned());
        assert!(reject_inert_options(&config).is_err());

        let mut config = RunConfig::default();
        config.compatibility.strict_reference = true;
        assert!(reject_inert_options(&config).is_err());
    }
}
