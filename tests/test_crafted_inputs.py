"""Crafted label and GeoTIFF files that are small on disk but would take far more memory
or time than their size justifies are refused quickly, within bounded memory.

Each read runs in a fresh interpreter whose CPU time is capped and whose resident memory
is watched, so a regression fails the test instead of exhausting the machine (Linux
only)."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the subprocess is watched through /proc"
)

#: Resident memory and CPU time each read may use: far below what the crafted files
#: asked for before they were refused, well above a normal read of the same files.
#: Resident memory, not address space: thread stacks and allocator arenas reserve
#: address space that varies from machine to machine.
_MEMORY_BYTES = 1536 << 20
_CPU_SECONDS = 30


def _resident_bytes(pid: int) -> int:
    """Resident memory of process ``pid``, 0 once it has exited."""
    try:
        resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return 0
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _bounded(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with capped CPU time, killing it if its resident
    memory passes ``_MEMORY_BYTES``."""
    limits = (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({_CPU_SECONDS}, {_CPU_SECONDS}))\n"
    )
    command = [sys.executable, "-c", limits + textwrap.dedent(code)]
    deadline = time.monotonic() + 4 * _CPU_SECONDS
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        peak = 0
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.01)
                break
            except subprocess.TimeoutExpired:
                peak = max(peak, _resident_bytes(process.pid))
                if peak > _MEMORY_BYTES or time.monotonic() > deadline:
                    process.kill()
                    stdout, stderr = process.communicate()
                    stderr += f"\nkilled at {peak >> 20} MiB resident"
                    break
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _refused(code: str, message: str) -> None:
    """``code`` raises ValueError containing ``message``, within the limits."""
    wrapped = f"""
        try:
{textwrap.indent(textwrap.dedent(code), " " * 12)}
        except ValueError as exc:
            print("refused:", exc)
        else:
            print("accepted")
    """
    result = _bounded(wrapped)
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr[-2000:]}"
    assert "refused:" in result.stdout, result.stdout
    assert message in result.stdout, result.stdout


# --- GeoTIFF -------------------------------------------------------------------------


def test_geotiff_repeating_one_tag_reads_it_once(tmp_path: Path) -> None:
    # One 4 MiB value that an IFD lists 4096 times: reading every entry took 16 GiB.
    # The first is kept, as libtiff does, and the file then fails on its missing tags.
    value = 4 << 20
    path = tmp_path / "repeat.tif"
    with path.open("wb") as f:
        f.write(b"II" + struct.pack("<HI", 42, 8 + value))
        f.seek(8 + value - 1)
        f.write(b"\0")
        f.write(struct.pack("<H", 4096))
        f.write(struct.pack("<HHII", 273, 1, value, 8) * 4096)
        f.write(struct.pack("<I", 0))
    _refused(
        f"""
        from mapcv.geotiff import GeoTiff
        GeoTiff({str(path)!r})
        """,
        "ImageWidth",
    )


def test_geotiff_ifds_sharing_one_large_value_are_refused(tmp_path: Path) -> None:
    # 1024 IFDs that each point at the same 4 MiB value: 4 GiB of tag data.
    value = 4 << 20
    path = tmp_path / "shared.tif"
    with path.open("wb") as f:
        f.write(b"II" + struct.pack("<HI", 42, 8 + value))
        f.seek(8 + value - 1)
        f.write(b"\0")
        position = 8 + value
        for index in range(1024):
            position += 2 + 12 + 4
            f.write(struct.pack("<HHHII", 1, 273, 1, value, 8))
            f.write(struct.pack("<I", 0 if index == 1023 else position))
    _refused(
        f"""
        from mapcv.geotiff import GeoTiff
        GeoTiff({str(path)!r})
        """,
        "hold more data than the file itself",
    )


# --- KML -----------------------------------------------------------------------------


def _kml(names: list[str]) -> bytes:
    placemark = (
        '<Placemark><ExtendedData><Data name="{name}"><value>v{index}</value></Data>'
        "</ExtendedData><Polygon><outerBoundaryIs><LinearRing><coordinates>0,0 1,0 1,1 0,0"
        "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>\n"
    )
    body = "".join(placemark.format(name=n, index=i % 3) for i, n in enumerate(names))
    return f'<?xml version="1.0"?><kml><Document>{body}</Document></kml>'.encode()


def test_inspecting_a_kml_with_many_field_names_is_linear(tmp_path: Path) -> None:
    # 3000 distinct field names: one parse of the whole file per name took minutes.
    (tmp_path / "labels.kml").write_bytes(_kml([f"f{i}" for i in range(3000)]))
    result = _bounded(
        f"""
        from pathlib import Path
        from mapcv.agent_tools import Sandbox, ToolState, inspect_labels
        from mapcv.cli import label_fields
        print(inspect_labels(ToolState(Sandbox(Path({str(tmp_path)!r}))), "labels.kml").summary)
        print(len(label_fields(Path({str(tmp_path)!r}) / "labels.kml")), "wizard fields")
        """
    )
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr[-2000:]}"
    assert "3,000 feature(s), 3000 field(s)" in result.stdout, result.stdout
    assert "3000 wizard fields" in result.stdout, result.stdout


def test_kml_field_scan_counts_each_field(tmp_path: Path) -> None:
    from mapcv.agent_tools import _scan_kml
    from mapcv.cli import label_fields

    data = _kml(["kind", "kind", "kind", "other"]).replace(
        b"</Document>",
        b'<Placemark><ExtendedData><SchemaData><SimpleData name="kind">v0</SimpleData>'
        b"</SchemaData></ExtendedData><Point><coordinates>0,0</coordinates></Point>"
        b"</Placemark></Document>",
    )
    scan = _scan_kml(data)
    assert scan.fields == {
        "kind": Counter({"v0": 1, "v1": 1, "v2": 1}),
        "other": Counter({"v0": 1}),
    }
    assert scan.with_field == Counter({"kind": 3, "other": 1})
    (tmp_path / "l.kml").write_bytes(data)
    assert label_fields(tmp_path / "l.kml") == {"kind": ["v0", "v1", "v2"], "other": ["v0"]}
