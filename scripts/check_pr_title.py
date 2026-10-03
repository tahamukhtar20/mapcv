"""Validate a pull request title against mapcv's Conventional Commit policy."""

from __future__ import annotations

import re
import sys


TITLE_PATTERN = re.compile(
    r"^(?:feat|fix|perf|refactor|docs|ci|chore|test)"
    r"(?:\([a-z0-9][a-z0-9._/-]*\))?!?: .+"
)


def main() -> int:
    """Return zero when the provided PR title follows the repository policy."""
    if len(sys.argv) != 2:
        print("usage: check_pr_title.py '<pull request title>'", file=sys.stderr)
        return 2

    title = sys.argv[1].strip()
    if TITLE_PATTERN.fullmatch(title):
        return 0

    print(
        "PR title must use Conventional Commits, for example 'feat(zarr): add EOPF imagery input'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
