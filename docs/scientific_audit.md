# Scientific implementation audit

This branch was audited against the single-master workflow in
[`dbekaert/StaMPS`](https://github.com/dbekaert/StaMPS) at revision
`c159eb81b16c446e0e8fdef7dd435eb22e0240ed`. Stage 6 flow semantics were also
checked against the Stanford SNAPHU 2.0.7 source. SNAPHU is an audit reference,
not a runtime dependency.

## Correctness fixes

| Area | Wrong-output risk | Implemented correction |
| --- | --- | --- |
| SNAP metadata | Stage 1 could use RSLC geometry while later stages used stale `heading` or `lambda` | SNAP preparation now derives and writes both values from the master RSLC; invalid lon/lat input fails before output |
| Stage 2 incidence | Native SNAP incidence could receive the legacy `+0.052` correction twice | Exact incidence now precedes the legacy look-angle fallback |
| Stage 3 selection | A null source interferogram could be missed after IFG dropping; incomplete probability caches could be fabricated | Source validity is checked on the full stack and required cache vectors are validated |
| Stage 5 merge | Missing overlap ownership could permit ambiguous multi-patch output | Every multi-patch input must provide `patch_noover.in` before merge I/O begins |
| Stage 6 flow | Each negative cycle was advanced once and counted once per unit, while upstream saturates it before one pivot count | Native Rust now saturates each cycle, preserves upstream flow-threshold rounding/order, and rejects zero-net opposing-edge cycles |
| Stage 6 caches | Parseable corruption or changed cache algorithms could reuse stale preprocessing | Grid, interpolation, space-time, and per-IFG checkpoints now carry schemas, fingerprints, and payload checksums |
| Stage 6 SCLA feedback | Requested deramping could silently omit a missing or malformed ramp | `scla_deramp='y'` requires a correctly shaped `ph_ramp` |
| Stage 7 regression | Regular acquisition cadence made the design rank-deficient and could zero the master baseline incorrectly | Only the redundant time column is removed; the weighted fit and exact master baseline are retained |
| Configuration | Accepted but inert tuning fields implied behavior that did not exist | Unsupported/no-op fields and external solver selections fail during configuration loading |
| Dataset boundary | `patch.list` or a linked patch directory could redirect stage writes outside the dataset root | Absolute/multi-component entries and patch symlinks or junctions are rejected after canonicalization |

Unsupported scientific branches are rejected before publication rather than
being approximated silently. The validated path is the available
single-master workflow. See [native_runtime.md](native_runtime.md) for the
current fail-closed list.

## Native runtime boundary

The installed `rustamps` binary and all production stages are Rust. The Cargo
normal/build graph has no Python, PyO3, NumPy, SNAPHU, Triangle, CMake,
`pkg-config`, or system-HDF5 dependency. Runtime Rust sources contain no
external process execution, and the macOS release binary links only
CoreFoundation and `libSystem`.

The retained Python tree and `oracle/pyproject.toml` are source-only developer
oracle material. They are not packaged, imported, or executed by the Rust
binary.

## Validation evidence

- Core and pipeline regressions cover cycle saturation, opposing-edge cycle
  rejection, upstream ties-to-even flow thresholds, checkpoint corruption,
  path containment, sensor metadata replacement, malformed SCLA feedback, and
  sequential/parallel determinism.
- A fresh 64-PS native Stage 6–8 run passed strict verification across 27
  artifacts against the fresh historical implementation output.
- On the 77,850-PS, 76-interferogram test dataset, the final converged 200 m
  profile now completes Stage 6 in `254.48 s` with four adaptive workers and
  strict-identical output. Peak RSS was `727,875,584` bytes.
- Controlled one/two/four-worker runs took `813.72`, `462.60`, and `254.48 s`;
  the scheduling-only setting is fingerprint-neutral and bitwise deterministic.
- The pathological interferogram that previously remained active after
  `780.57 s` completed in `9.85 s` after the upstream cycle-saturation fix.
- The completed full tree passed strict comparison across 36 artifacts against
  the prior converged implementation state.

The faster profiles change grid resolution, not flow convergence. Their
scientific tolerances are explicit, bounded, and intended to be checked against
the strict profile before use on a new dataset.
