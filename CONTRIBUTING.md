# Contributing to mapcv

Thank you for your interest in contributing to **mapcv**! We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code changes.

This document outlines the process for contributing to the project and setting up your local development environment.

## Getting Started

### 1. Development Environment Setup
`mapcv` is a hybrid Python and Rust project. We use `uv` for Python dependency management and `maturin` to build the Rust extensions.

Make sure you have Rust and Python 3.10+ installed. First, fork and clone the repository, then navigate to the project directory and install all development dependencies:

```bash
uv sync --extra dev
uv run maturin develop
```

> **Note**: If you modify the Rust code (`src/*.rs`), you must re-run `uv run maturin develop` for the changes to take effect in Python.

### 2. Pre-commit Hooks
`pre-commit` hooks run `cargo fmt`, `clippy`, `ruff`, `mypy`, and `pytest` automatically before every commit.

Install `pre-commit` if you don't have it, then install the hooks:
```bash
pip install pre-commit
pre-commit install
```

## Running Tests and Linters Manually

You can manually trigger the linters and tests to verify your changes before committing:

**Python Formatting & Linting:**
```bash
uv run ruff format .
uv run ruff check .
```

**Python Type Checking:**
```bash
uv run mypy --strict .
```

**Rust Formatting & Linting:**
```bash
cargo fmt
cargo clippy
```

**Running Tests:**
```bash
uv run pytest
```

## Coding Conventions

### Python

#### Files
- All source files live under `src/mapcv/` and are named `snake_case.py`.
- Test files live under `tests/` and are named `test_<module>.py`.

#### Module header
Every `.py` file starts with, in order:
1. A one-line module docstring - plain prose, ends with `.`
2. `from __future__ import annotations`
3. Imports grouped as: stdlib -> third-party -> local, each group separated by a blank line.

#### Type annotations
- Required on every public function and method signature (parameters + return).
- Use `typing` module forms: `Dict`, `List`, `Optional`, `Tuple`, `Literal` - not PEP 585
  bare generics (`dict`, `list`). The `from __future__ import annotations` already enables
  deferred evaluation; keep the import style consistent.

#### Docstrings
- **Module**: one-line, plain prose.
- **Public class**: one-line, plain prose.
- **Public function / method**: one-line minimum. Add a second paragraph only when the
  behaviour is non-obvious (side effects, edge cases, algorithm sketch).
- **Private function / method (`_` prefix)**: omit unless the logic is non-obvious.
- Format: plain prose (not NumPy / Google style). No section headers (`Args:`, `Returns:`).
  Describe parameters inline in the body text when they need explanation.

#### Inline comments
- Only for *why*, never *what*. A comment that re-states the code is noise.
- Single `#` with one space. Placed on its own line above the statement it refers to.

#### Linting / formatting
- `ruff` enforces line length (100) and import order.
- `mypy --strict` must pass with zero errors.
- No `# type: ignore` unless truly unavoidable; prefer a cast or a narrower annotation.

### Rust

#### Files
- Source files live under `src/` and are named `snake_case.rs`.
- Each file is a module; `lib.rs` is the PyO3 entry point.

#### Module header
Every `.rs` file starts with `//!` inner doc comment - one line minimum describing
the module's responsibility.

#### Doc comments
- **Public items** (`pub fn`, `pub struct`, `pub enum`, `pub type`, `pub const`):
  `///` doc comment required. Include `# Errors` and/or `# Panics` sections where
  applicable.
- **Private helpers**: `///` doc comment recommended whenever the function contains
  non-trivial logic; omit for trivial one-liners.
- Keep doc comments factual and concise. No restating of the function signature.

#### Inline comments
- `//` with one space. For *why* only.
- When suppressing a Clippy lint (`#[allow(clippy::...)]`), a `//` comment on the
  preceding line must explain the reason.

#### Error handling
- Public functions return `Result<T, String>` (propagated to Python as `PyErr`).
- Use `?` with `.map_err(|e| e.to_string())` for error conversion.
- No `.unwrap()` or `.expect()` in public functions. `.expect()` is allowed in private
  helpers only when the invariant genuinely cannot be violated; document it with a
  comment.
- `# Panics` section in the doc comment whenever a function can panic.

#### Miscellaneous
- Mark pure functions `#[must_use]`.
- Avoid magic numbers; extract named constants (`const TILE_PX: usize = 256`).
- `pub(crate)` for items used across modules but not part of the Python API.

## How to Submit a Contribution

1. **Open an Issue:** If you're planning a significant change, please open an issue first to discuss it with the maintainers.
2. **Create a Branch:** Create a feature branch (`git checkout -b feat/your-feature-name`).
3. **Commit your Changes:** Make your changes, ensuring that all `pre-commit` checks pass.
4. **Submit a Pull Request:** Push your branch to your fork and submit a Pull Request against our `main` branch.

Please ensure your PR description clearly describes the problem and the proposed solution. Include references to any related issues.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainers.
