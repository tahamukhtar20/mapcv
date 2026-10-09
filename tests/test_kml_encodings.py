"""KML label files in the encoding their byte order mark or XML declaration names."""

from __future__ import annotations

from pathlib import Path

import pytest

from mapcv.labels import kml_to_utf8, load_vector_labels, parse_kml

_KML = (
    '<?xml version="1.0" encoding="{enc}"?>'
    '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
    '<ExtendedData><Data name="kind"><value>{value}</value></Data></ExtendedData>'
    "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
    "4.00,52.00 4.01,52.00 4.01,52.01 4.00,52.00"
    "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>"
)


def _kml(encoding: str, value: str = "café €") -> str:
    return _KML.format(enc=encoding, value=value)


@pytest.mark.parametrize(
    ("declared", "data"),
    [
        ("UTF-8", _kml("UTF-8").encode("utf-8")),
        ("UTF-8", b"\xef\xbb\xbf" + _kml("UTF-8").encode("utf-8")),
        ("UTF-16", b"\xff\xfe" + _kml("UTF-16").encode("utf-16-le")),
        ("UTF-16", b"\xfe\xff" + _kml("UTF-16").encode("utf-16-be")),
        ("UTF-16LE", _kml("UTF-16LE").encode("utf-16-le")),
        ("UTF-16BE", _kml("UTF-16BE").encode("utf-16-be")),
        ("UTF-32", b"\xff\xfe\x00\x00" + _kml("UTF-32").encode("utf-32-le")),
        ("windows-1252", _kml("windows-1252").encode("cp1252")),
        ("ISO-8859-1", _kml("ISO-8859-1", "café").encode("latin-1")),
        ("iso-8859-1", _kml("iso-8859-1", "café").encode("latin-1")),
    ],
    ids=[
        "utf8",
        "utf8-bom",
        "utf16-le-bom",
        "utf16-be-bom",
        "utf16le-no-bom",
        "utf16be-no-bom",
        "utf32-bom",
        "cp1252",
        "latin1",
        "latin1-lowercase",
    ],
)
def test_a_kml_file_is_read_in_the_encoding_it_declares(declared: str, data: bytes) -> None:
    geometries, classes = parse_kml(data, "kind")
    assert len(geometries) == 1
    expected = "café" if declared.upper().startswith("ISO") else "café €"
    assert classes == {expected: 1}
    box = geometries[0][0].bounds
    assert box == pytest.approx((4.0, 52.0, 4.01, 52.01))
    # Transcoding is stable: the result says UTF-8 and goes through again unchanged.
    once = kml_to_utf8(data)
    assert kml_to_utf8(once) == once


def test_a_kml_file_in_an_unsupported_or_wrong_encoding_says_so() -> None:
    sjis = _kml("Shift_JIS", "町").encode("shift_jis")
    with pytest.raises(
        ValueError, match=r"encoding 'Shift_JIS' is not supported; save it as UTF-8"
    ):
        parse_kml(sjis, "kind")
    bad = _kml("windows-1252", "x").encode("utf-8").replace(b"x<", b"\x81<")
    with pytest.raises(ValueError, match=r"'windows-1252' but has bytes that are not valid"):
        parse_kml(bad, "kind")
    # Declares UTF-16 but is UTF-8 with non-ASCII text: not UTF-16, not readable as it says.
    mislabelled = _kml("UTF-16").encode("utf-8").replace(b"caf", b"\xe9af")
    with pytest.raises(ValueError, match=r"is not saved as UTF-16 text"):
        parse_kml(mislabelled, "kind")
    # Declares UTF-16 but is plain UTF-8: read as UTF-8, as before.
    assert parse_kml(_kml("UTF-16").encode("utf-8"), "kind")[1] == {"café €": 1}


def test_a_kml_label_file_in_utf16_loads_through_load_vector_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.kml"
    path.write_bytes(b"\xff\xfe" + _kml("UTF-16").encode("utf-16-le"))
    geometries, classes = load_vector_labels(path, "kind")
    assert len(geometries) == 1 and classes == {"café €": 1}


def test_the_label_inspectors_read_a_utf16_kml_too(tmp_path: Path) -> None:
    from mapcv.agent_tools import _scan_kml
    from mapcv.cli import label_fields

    data = b"\xff\xfe" + _kml("UTF-16").encode("utf-16-le")
    path = tmp_path / "labels.kml"
    path.write_bytes(data)
    assert label_fields(path) == {"kind": ["café €"]}
    scan = _scan_kml(data)
    assert scan.features == 1 and dict(scan.fields["kind"]) == {"café €": 1}
