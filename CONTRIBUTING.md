# Contributing to Rustamps

## Repository setup

```bash
git clone https://github.com/sirbastiano/rustamps.git
cd rustamps
cargo build --release --locked
cargo test --workspace --locked
```

## Code standards

- Keep the production implementation and dependency graph native Rust.
- Preserve StaMPS MAT contracts and reject unsupported scientific modes.
- Keep behavior deterministic where feasible and document inferred behavior in implementation.

## PR workflow

1. Implement a small, scoped change.
2. Run local static checks and project tests.
3. Include reproducible commands or examples in the PR description.
4. Run strict or scientific verifier checks when touching stage math.

The retained Python tree and `oracle/pyproject.toml` are optional historical
oracle tooling. They are not part of production setup or deployment.

## Branching model

Default model is short-lived feature branches with one purpose per branch.
Merge after passing quality checks and audit gates.
