# mapcv Coding Conventions

## Python

### Files
- All source files live under `src/mapcv/` and are named `snake_case.py`.
- Test files live under `tests/` and are named `test_<module>.py`.

### Module header
Every `.py` file starts with, in order:
1. A one-line module docstring - plain prose, ends with `.`
2. `from __future__ import annotations`
3. Imports grouped as: stdlib -> third-party -> local, each group separated by a blank line.

### Type annotations
- Required on every public function and method signature (parameters + return).
- Use `typing` module forms: `Dict`, `List`, `Optional`, `Tuple`, `Literal` - not PEP 585
  bare generics (`dict`, `list`). The `from __future__ import annotations` already enables
  deferred evaluation; keep the import style consistent.

### Docstrings
- **Module**: one-line, plain prose.
- **Public class**: one-line, plain prose.
- **Public function / method**: one-line minimum. Add a second paragraph only when the
  behaviour is non-obvious (side effects, edge cases, algorithm sketch).
- **Private function / method (`_` prefix)**: omit unless the logic is non-obvious.
- Format: plain prose (not NumPy / Google style). No section headers (`Args:`, `Returns:`).
  Describe parameters inline in the body text when they need explanation.

### Inline comments
- Only for *why*, never *what*. A comment that re-states the code is noise.
- Single `#` with one space. Placed on its own line above the statement it refers to.

### Linting / formatting
- `ruff` enforces line length (100) and import order.
- `mypy --strict` must pass with zero errors.
- No `# type: ignore` unless truly unavoidable; prefer a cast or a narrower annotation.

---

## Rust

### Files
- Source files live under `src/` and are named `snake_case.rs`.
- Each file is a module; `lib.rs` is the PyO3 entry point.

### Module header
Every `.rs` file starts with `//!` inner doc comment - one line minimum describing
the module's responsibility.

### Doc comments
- **Public items** (`pub fn`, `pub struct`, `pub enum`, `pub type`, `pub const`):
  `///` doc comment required. Include `# Errors` and/or `# Panics` sections where
  applicable.
- **Private helpers**: `///` doc comment recommended whenever the function contains
  non-trivial logic; omit for trivial one-liners.
- Keep doc comments factual and concise. No restating of the function signature.

### Inline comments
- `//` with one space. For *why* only.
- When suppressing a Clippy lint (`#[allow(clippy::...)]`), a `//` comment on the
  preceding line must explain the reason.

### Error handling
- Public functions return `Result<T, String>` (propagated to Python as `PyErr`).
- Use `?` with `.map_err(|e| e.to_string())` for error conversion.
- No `.unwrap()` or `.expect()` in public functions. `.expect()` is allowed in private
  helpers only when the invariant genuinely cannot be violated; document it with a
  comment.
- `# Panics` section in the doc comment whenever a function can panic.

### Miscellaneous
- Mark pure functions `#[must_use]`.
- Avoid magic numbers; extract named constants (`const TILE_PX: usize = 256`).
- `pub(crate)` for items used across modules but not part of the Python API.
