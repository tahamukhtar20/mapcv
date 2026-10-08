"""Reading a Zarr store never runs code from it (numcodecs' ``pickle`` codec)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
numcodecs = pytest.importorskip("numcodecs")

from mapcv._zarr_safety import array_problem, check_group, pickle_codec_refused


class _Payload:
    """Writes ``marker`` when unpickled, as an attacker's object would run code."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> Any:
        return (Path.write_text, (self.marker, "code ran"))


def _manifest() -> dict[str, Any]:
    return {
        "version": 3,
        "task": "segmentation",
        "sources": [
            {
                "name": "image",
                "crs": "EPSG:3857",
                "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                "patch_shape": [256, 256, 3],
            }
        ],
        "patches": [
            {
                "files": {"image": "Images/patch_0000000.png"},
                "row": 0,
                "col": 0,
                "padded": False,
                "chunk": 0,
                "summary": {},
            }
        ],
    }


def _store(root: Path, marker: Path) -> Path:
    """A store laid out like ``mapcv export -f zarr`` whose image array unpickles."""
    group = zarr.open_group(str(root), mode="w")
    images = group.require_group("images")
    array = images.create_dataset(
        "image", shape=(1,), chunks=(1,), dtype=object, object_codec=numcodecs.Pickle()
    )
    data = np.empty(1, dtype=object)
    data[0] = _Payload(marker)
    array[:] = data
    for name in ("split", "row", "col"):
        group.create_dataset(name, data=np.zeros(1, dtype=np.int64))
    (root / "manifest.json").write_text(json.dumps(_manifest()))
    return root


def test_a_store_that_would_unpickle_is_refused(tmp_path: Path) -> None:
    from mapcv.data import MapcvDataset

    marker = tmp_path / "PWNED"
    store = _store(tmp_path / "store.zarr", marker)
    with pytest.raises(ValueError, match="'images/image' holds '|O' values, not numbers"):
        dataset = MapcvDataset(store, split="all", as_tensors=False)
        dataset[0]
    assert not marker.exists()


def _rewrite(path: Path, **fields: Any) -> None:
    meta = json.loads(path.read_text())
    meta.update(fields)
    path.write_text(json.dumps(meta))


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"filters": [{"id": "pickle"}]}, "codec 'pickle'"),
        ({"compressor": {"id": "msgpack2"}}, "codec 'msgpack2'"),
        ({"compressor": {"id": "made-up"}}, "codec 'made-up'"),
        ({"dtype": "|O", "filters": [{"id": "vlen-utf8"}]}, "not numbers"),
        ({"dtype": "<U8"}, "not numbers"),
        ({"dtype": [["a", "<i4"]]}, "structured data type"),
        ({"filters": "pickle"}, "not a list"),
    ],
)
def test_arrays_other_than_plain_numbers_are_refused(
    tmp_path: Path, fields: dict[str, Any], problem: str
) -> None:
    group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    group.require_group("masks").create_dataset("m", data=np.zeros((2, 4), dtype=np.uint8))
    _rewrite(tmp_path / "s.zarr" / "masks" / "m" / ".zarray", **fields)
    with pytest.raises(ValueError, match=problem):
        check_group(zarr.open_group(str(tmp_path / "s.zarr"), mode="r"), "s.zarr")


def test_what_mapcv_writes_is_accepted(tmp_path: Path) -> None:
    group = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    images = group.require_group("images")
    for dtype in (np.uint8, np.uint16, np.int16, np.float32, np.float64, np.bool_):
        images.create_dataset(np.dtype(dtype).name, data=np.zeros((2, 3, 4), dtype=dtype))
    group.create_dataset("split", data=np.zeros(2, dtype=np.uint8), compressor=None)
    group.create_dataset(
        "row", data=np.zeros(2, dtype=np.int64), filters=[numcodecs.Delta(dtype="<i8")]
    )
    check_group(zarr.open_group(str(tmp_path / "s.zarr"), mode="r"), "s.zarr")
    assert array_problem({"dtype": "<f4", "compressor": {"id": "zstd", "level": 3}}) is None


def test_the_pickle_codec_is_refused_only_inside_the_block(tmp_path: Path) -> None:
    marker = tmp_path / "PWNED"
    store = _store(tmp_path / "store.zarr", marker)
    with pickle_codec_refused(), pytest.raises(zarr.errors.MetadataError) as excinfo:
        zarr.open_group(str(store), mode="r")["images/image"][0]
    assert "'pickle' codec" in str(excinfo.value.__cause__)
    assert not marker.exists()
    with pickle_codec_refused(), pickle_codec_refused():
        pass
    assert isinstance(numcodecs.get_codec({"id": "pickle"}), numcodecs.Pickle)
