from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CARGO = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
CARGO_LOCK = (REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8")
MAKEFILE = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
ORACLE_PROJECT = (REPO_ROOT / "oracle" / "pyproject.toml").read_text(encoding="utf-8")
ENVIRONMENT = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
GITIGNORE = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
NATIVE_DOC = (REPO_ROOT / "docs" / "native_runtime.md").read_text(encoding="utf-8")
RELEASE_DOC = (REPO_ROOT / "docs" / "release.md").read_text(encoding="utf-8")
DOC_INDEX = (REPO_ROOT / "docs" / "index.html").read_text(encoding="utf-8")


def test_cargo_root_is_the_only_installable_product() -> None:
    assert "autolib = false" in CARGO
    assert "autobins = false" in CARGO
    assert 'name = "pystamps"' in CARGO
    assert 'path = "crates/pystamps-cli/src/main.rs"' in CARGO
    assert "pyo3" not in CARGO.lower()
    assert "numpy" not in CARGO.lower()


def test_locked_production_graph_excludes_runtime_bridges() -> None:
    package_names = set(re.findall(r'^name = "([^"]+)"$', CARGO_LOCK, flags=re.MULTILINE))
    forbidden = {
        "hdf5",
        "hdf5-sys",
        "numpy",
        "pyo3",
        "pyo3-build-config",
        "pyo3-ffi",
        "pyo3-macros",
        "pyo3-macros-backend",
    }
    assert package_names.isdisjoint(forbidden)
    assert "hdf5-pure" in package_names


def test_python_oracle_is_a_non_installable_dev_environment() -> None:
    assert not (REPO_ROOT / "pyproject.toml").exists()

    assert 'name = "pystamps-oracle-dev"' in ORACLE_PROJECT
    assert "[dependency-groups]" in ORACLE_PROJECT
    assert "[tool.uv]" in ORACLE_PROJECT
    assert "package = false" in ORACLE_PROJECT
    assert "[tool.pytest.ini_options]" in ORACLE_PROJECT

    forbidden = (
        "[build-system]",
        "[project.scripts]",
        "setuptools",
        "cibuildwheel",
        "twine",
    )
    for marker in forbidden:
        assert marker not in ORACLE_PROJECT

    for obsolete in ("pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in", "uv.lock"):
        assert not (REPO_ROOT / obsolete).exists()

    dist = REPO_ROOT / "dist"
    assert not list(dist.glob("*.whl"))
    assert not list(dist.glob("*.tar.gz"))
    assert "/dist/" in GITIGNORE
    assert "!dist/" not in GITIGNORE


def test_makefile_separates_cargo_product_from_oracle_checks() -> None:
    assert "test:\n\t$(CARGO) test --workspace --locked" in MAKEFILE
    assert "build:\n\t$(CARGO) build --release --locked" in MAKEFILE
    assert "$(CARGO) run --release --locked -- verify" in MAKEFILE
    assert "oracle-setup:" in MAKEFILE
    assert "oracle-test:" in MAKEFILE
    assert "oracle-audit:" in MAKEFILE
    assert "oracle-verify:" in MAKEFILE
    assert "PYTHONPATH=." in MAKEFILE

    assert "python setup.py" not in MAKEFILE
    assert "python -m build" not in MAKEFILE
    assert "twine" not in MAKEFILE
    assert "uv run pystamps" not in MAKEFILE


def test_conda_file_is_oracle_only_not_a_build_surface() -> None:
    assert "name: pystamps-oracle" in ENVIRONMENT
    assert "python=3.12" in ENVIRONMENT
    assert "setuptools-rust" not in ENVIRONMENT
    assert "wheel" not in ENVIRONMENT


def test_current_docs_define_native_install_and_release() -> None:
    assert "Standalone Rust implementation" in README
    assert "cargo install --path ." in README
    assert "No Python environment or system HDF5 library is required" in README
    assert "tool.uv.package = false" in README

    assert "does not load Python" in NATIVE_DOC
    assert "defines no Python build backend" in NATIVE_DOC
    assert "make oracle-*" in NATIVE_DOC

    assert "Standalone Rust runtime" in DOC_INDEX
    assert "cargo install --path . --locked" in DOC_INDEX
    assert "Historical oracle material" in DOC_INDEX
    assert "uv run pystamps" not in DOC_INDEX

    assert "# Native release process" in RELEASE_DOC
    assert "cargo test --workspace --locked" in RELEASE_DOC
    assert "cargo build --release --locked" in RELEASE_DOC
    assert "cargo install --path . --locked" in RELEASE_DOC
    assert "make oracle-audit" in RELEASE_DOC
    assert "pip install" not in RELEASE_DOC
    assert "cibuildwheel" not in RELEASE_DOC
    assert "twine" not in RELEASE_DOC


def test_native_config_rejects_external_solver_documentation() -> None:
    assert "external or SNAPHU Stage 6 solver" in NATIVE_DOC
    assert "The native Stage 6 solver is self-contained" in README
    assert "never executes" in README
