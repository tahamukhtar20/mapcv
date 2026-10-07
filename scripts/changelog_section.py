"""Print the CHANGELOG.md section for one version, used as GitHub release notes."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def changelog_section(text: str, version: str) -> str:
    """Return the body under ``## [version]``, up to the next ``## [`` heading."""
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise ValueError(f"CHANGELOG.md has no section for version {version}")
    return match.group(1).strip() + "\n"


def main() -> int:
    """Print the section for ``sys.argv[1]``; exit non-zero when it is missing."""
    if len(sys.argv) != 2:
        print("usage: changelog_section.py <version>", file=sys.stderr)
        return 2
    try:
        print(changelog_section(Path("CHANGELOG.md").read_text(), sys.argv[1]), end="")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
