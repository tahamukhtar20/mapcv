"""Check that the installed mapcv came from a wheel in a directory, not from PyPI."""

from __future__ import annotations

import base64
import hashlib
import sys
import zipfile
from importlib import metadata
from pathlib import Path


def record_hash(data: bytes) -> str:
    """Return the hash of ``data`` as a wheel RECORD writes it (``sha256=<urlsafe b64>``)."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode()


def find_wheel(wheel_dir: Path, path: str, digest: str) -> Path | None:
    """Return the wheel in ``wheel_dir`` whose RECORD lists ``path`` with ``digest``."""
    for wheel in sorted(wheel_dir.glob("mapcv-*.whl")):
        with zipfile.ZipFile(wheel) as zf:
            record = next(n for n in zf.namelist() if n.endswith(".dist-info/RECORD"))
            for line in zf.read(record).decode().splitlines():
                name, hash_, _size = line.rsplit(",", 2)
                if name == path and hash_ == digest:
                    return wheel
    return None


def main() -> int:
    """Compare the installed native module with the wheels in ``sys.argv[1]``."""
    if len(sys.argv) != 2:
        print("usage: check_installed_wheel.py <wheel-dir>", file=sys.stderr)
        return 2
    dist = metadata.distribution("mapcv")
    # The compiled extension only: the package also ships its type stub, _mapcv_rs.pyi.
    native = [
        f
        for f in dist.files or []
        if f.name.startswith("_mapcv_rs") and f.suffix in (".so", ".pyd", ".dylib")
    ]
    if len(native) != 1:
        print(f"error: expected one _mapcv_rs module, found {native}", file=sys.stderr)
        return 1
    path = native[0].as_posix()
    digest = record_hash(Path(str(dist.locate_file(native[0]))).read_bytes())
    wheel = find_wheel(Path(sys.argv[1]), path, digest)
    if wheel is None:
        print(
            f"error: the installed {path} ({digest}) is in no wheel in {sys.argv[1]}; "
            "pip installed mapcv from somewhere else",
            file=sys.stderr,
        )
        return 1
    print(f"mapcv {dist.version} is installed from {wheel.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
