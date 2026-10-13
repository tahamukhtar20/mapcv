"""Crafted label and GeoTIFF files that are small on disk but would take far more memory
or time than their size justifies are refused quickly, within bounded memory.

Each read runs in a fresh interpreter whose CPU time is capped and whose resident memory
is watched, so a regression fails the test instead of exhausting the machine (Linux
only)."""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

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


def _bounded(code: str, memory: int = _MEMORY_BYTES) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with capped CPU time, killing it if its resident
    memory passes ``memory`` bytes."""
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
                if peak > memory or time.monotonic() > deadline:
                    process.kill()
                    stdout, stderr = process.communicate()
                    stderr += f"\nkilled at {peak >> 20} MiB resident"
                    break
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _refused(code: str, message: str) -> str:
    """``code`` raises ValueError containing ``message``, within the limits; its output."""
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
    return result.stdout


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


# --- GeoParquet ----------------------------------------------------------------------

_GEO = {
    "version": "1.0.0",
    "primary_column": "geometry",
    "columns": {"geometry": {"encoding": "WKB"}},
}
_POINT = bytes.fromhex("0101000000000000000000f03f000000000000f03f")


def _parquet(path: Path, rows: int, value: str, *, dictionary: bool) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    schema = pa.schema(
        [("geometry", pa.binary()), ("class", pa.large_string())],
        metadata={b"geo": json.dumps(_GEO).encode()},
    )
    one = pa.array([value], pa.large_string())
    with pq.ParquetWriter(
        path, schema, compression="zstd", compression_level=19, use_dictionary=dictionary
    ) as writer:
        if dictionary:  # one row group; the dictionary holds the value once
            column = pa.chunked_array([one] * rows)
            writer.write_table(pa.table({"geometry": [_POINT] * rows, "class": column}, schema))
        else:  # one row per row group, each value compressed on its own
            for _ in range(rows):
                writer.write_table(pa.table({"geometry": [_POINT], "class": one}, schema))


def test_geoparquet_decompressing_to_gigabytes_is_refused(tmp_path: Path) -> None:
    # 12 x 64 MiB of text compressed to a few KiB.
    path = tmp_path / "plain.parquet"
    _parquet(path, 12, "x" * (64 << 20), dictionary=False)
    assert path.stat().st_size < 1 << 20
    _refused(
        f"""
        from mapcv.vector_files import read_geoparquet
        read_geoparquet(__import__("pathlib").Path({str(path)!r}), ["class"])
        """,
        "its columns decode to more than",
    )


def test_the_geoparquet_refusal_gives_its_limit_in_one_unit_and_a_fitting_next_step(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plain.parquet"
    _parquet(path, 12, "x" * (64 << 20), dictionary=False)
    size = path.stat().st_size
    code = """
        from mapcv.vector_files import read_geoparquet
        read_geoparquet(__import__("pathlib").Path({path!r}), {fields})
        """
    # A column is named: the hint to name one is not repeated back.
    named = _refused(code.format(path=str(path), fields='["class"]'), "its columns decode to more")
    limit = (64 << 20) + 64 * size
    assert f"more than {limit / (1 << 20):.1f} MiB, the limit for a file of " in named
    assert "(64 MiB plus 64 times its size)" in named
    assert "MB" not in named and "labels.label_field" not in named
    assert "write it without the large columns and try again" in named
    # No column named (the wizard's read of every column): naming one is the way out.
    unnamed = _refused(code.format(path=str(path), fields="None"), "its columns decode to more")
    assert "(or name the one column needed in labels.label_field)" in unnamed


def test_geoparquet_dictionary_expanding_to_gigabytes_is_refused(tmp_path: Path) -> None:
    # A 512 KiB dictionary value used by 20,000 rows: 10 GiB once expanded, while the
    # metadata states only the dictionary and the indices.
    path = tmp_path / "dictionary.parquet"
    _parquet(path, 20_000, "x" * (512 << 10), dictionary=True)
    _refused(
        f"""
        from mapcv.vector_files import read_geoparquet
        read_geoparquet(__import__("pathlib").Path({str(path)!r}), ["class"])
        """,
        "its columns decode to more than",
    )


def test_geoparquet_dictionary_columns_read_as_their_values(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from mapcv.vector_files import read_geoparquet

    classes = ["roof", None, "road", "roof", ""]
    table = pa.table(
        {
            "geometry": pa.array([_POINT] * 5, pa.binary()),
            "class": pa.array(classes, pa.string()),
            "code": pa.array([b"a", b"b", None, b"a", b"c"], pa.binary()),
            "height": pa.array([1.5, 2.0, None, 4.0, 5.0]),
        }
    ).replace_schema_metadata({b"geo": json.dumps(_GEO).encode()})
    path = tmp_path / "labels.parquet"
    pq.write_table(table, path)  # dictionary-encoded by default
    assert "RLE_DICTIONARY" in pq.ParquetFile(path).metadata.row_group(0).column(1).encodings
    result = read_geoparquet(path, ["class", "code", "height"])
    assert result.columns == {
        "class": classes,
        "code": [b"a", b"b", None, b"a", b"c"],
        "height": [1.5, 2.0, None, 4.0, 5.0],
    }
    assert [g.wkb for g in result.geometries if g is not None] == [_POINT] * 5


def test_geoparquet_past_the_dictionary_bound_is_counted_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One long dictionary entry among short ones: values times the longest entry passes
    # the budget, so the text columns are read as dictionaries and counted as expanded.
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from mapcv import vector_files

    classes = ["x" * 4000] + ["roof", "road"] * 500
    table = pa.table(
        {"geometry": pa.array([_POINT] * 1001, pa.binary()), "class": classes}
    ).replace_schema_metadata({b"geo": json.dumps(_GEO).encode()})
    path = tmp_path / "labels.parquet"
    pq.write_table(table, path)
    stated = vector_files._parquet_stated_bytes(pq.ParquetFile(path), ["geometry", "class"])
    monkeypatch.setattr(vector_files, "_PARQUET_BYTES_PER_BYTE", 0)
    monkeypatch.setattr(vector_files, "_PARQUET_BYTES_FLOOR", stated + (1 << 20))
    seen: list[bool] = []
    counted = vector_files._decoded_bytes

    def spy(pa_module: Any, array: Any) -> int:
        seen.append(pa.types.is_dictionary(array.type))
        return counted(pa_module, array)

    monkeypatch.setattr(vector_files, "_decoded_bytes", spy)
    result = vector_files.read_geoparquet(path, ["class"])
    assert result.columns == {"class": classes}
    assert len(result.geometries) == 1001
    assert seen == [True, True], "both text columns read as dictionaries"


# --- GeoPackage ----------------------------------------------------------------------


def _gpkg_view(path: Path, select: str, key: str = "fid") -> None:
    """A GeoPackage whose only layer is a view of 1000 rows: ``select`` gives its class
    column, ``key`` names its first column."""
    db = sqlite3.connect(path)
    db.executescript(
        f"""
        CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT, srs_id INTEGER PRIMARY KEY,
          organization TEXT, organization_coordsys_id INTEGER, definition TEXT,
          description TEXT);
        INSERT INTO gpkg_spatial_ref_sys VALUES ('WGS 84', 4326, 'EPSG', 4326, 'undefined', NULL);
        CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT,
          identifier TEXT, description TEXT, last_change TEXT, min_x REAL, min_y REAL,
          max_x REAL, max_y REAL, srs_id INT);
        INSERT INTO gpkg_contents VALUES ('labels', 'features', 'labels', '', '', 0, 0, 1, 1, 4326);
        CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT,
          geometry_type_name TEXT, srs_id INTEGER, z TINYINT, m TINYINT);
        INSERT INTO gpkg_geometry_columns VALUES ('labels', 'geom', 'POLYGON', 4326, 0, 0);
        CREATE VIEW labels AS
          WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 1000)
          SELECT i AS {key}, NULL AS geom, {select} AS class FROM n;
        """
    )
    db.commit()
    db.close()


@pytest.mark.parametrize(
    ("select", "key"),
    [
        ("printf('%.*c', 4 * 1048576, 'x')", "fid"),
        ("zeroblob(4 * 1048576)", "fid"),
        # A view with a rowid column (or any view, where SQLite allows rowid in views)
        # was ordered by it, so SQLite sorted all its rows before returning the first.
        ("zeroblob(4 * 1048576)", "rowid"),
    ],
    ids=["text", "blob", "rowid"],
)
def test_geopackage_view_building_gigabytes_is_refused(
    tmp_path: Path, select: str, key: str
) -> None:
    # 1000 rows of a 4 MiB value computed by the view: 4 GiB from a 20 KiB file.
    path = tmp_path / "view.gpkg"
    _gpkg_view(path, select, key)
    _refused(
        f"""
        from mapcv.vector_files import read_gpkg
        read_gpkg(__import__("pathlib").Path({str(path)!r}), fields=["class"])
        """,
        "returns far more data than a file of this size holds",
    )


def test_geopackage_view_of_small_values_still_reads(tmp_path: Path) -> None:
    from mapcv.vector_files import read_gpkg

    path = tmp_path / "view.gpkg"
    _gpkg_view(path, "printf('class %d', i % 3)")
    table = read_gpkg(path, fields=["class"])
    assert Counter(table.columns["class"]) == {"class 0": 333, "class 1": 334, "class 2": 333}


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
