# Release Process

`pystamps` releases are manual and tag-driven. The preferred entry points are the `Makefile` targets in the repo root.

## Prerequisites

- Python 3.12+
- `uv`
- PyPI credentials available to `twine`
- a clean Git worktree
- local access to the validation datasets required by the parity audit

## Release Steps

1. Sync the maintainer environment:

   ```bash
   make sync
   ```

2. Run the test gate:

   ```bash
   make test
   ```

3. Run the strict parity gate:

   ```bash
   make audit
   ```

4. Create and push a release tag using the version form `vX.Y.Z`.

5. Build the release artifacts from the tagged commit:

   ```bash
   make build
   ```

6. Validate the built artifacts:

   ```bash
   make dist-check
   ```

7. Upload to TestPyPI for rehearsal when needed:

   ```bash
   make publish-testpypi
   ```

8. Upload the final artifacts to PyPI:

   ```bash
   make publish-pypi
   ```

## Release Requirements

- `pytest` must pass.
- `latest_audit.json` must report no failed parity workflows.
- `python -m build` must emit exactly one wheel and one sdist.
- `twine check` must pass on all files in `dist/`.

## Distribution Scope

- Published artifacts include the `pystamps` Python package and package metadata.
- Release artifacts do not include `inputs_and_outputs/`, `tmp/`, or the vendored `StaMPS/` tree.
- External binaries such as `triangle` and `snaphu` remain user-managed prerequisites.
