"""GeoPackages in WAL mode in read-only folders (a read-only mount, a file owned by someone else)."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from test_vector_labels import SQUARE, gpkg_blob, make_gpkg

from mapcv.labels import load_vector_labels


def _wal_gpkg(folder: Path, keep_wal: bool = False) -> tuple[Path, sqlite3.Connection | None]:
    path = make_gpkg(folder / "labels.gpkg", [(gpkg_blob(SQUARE), "a")])
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    if not keep_wal:
        connection.close()
        return path, None
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute(
        "INSERT INTO feat (geom, label) VALUES (?, ?)", (gpkg_blob(SQUARE), "unmerged")
    )
    connection.commit()
    return path, connection


@pytest.fixture
def read_only() -> Iterator[Callable[[Path], Path]]:
    folders: list[Path] = []

    def lock(folder: Path) -> Path:
        folder.chmod(0o555)
        folders.append(folder)
        return folder

    yield lock
    for folder in folders:
        folder.chmod(0o755)


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="a read-only folder only stops users that are not root",
)
def test_a_wal_geopackage_in_a_read_only_folder_is_read(
    tmp_path: Path, read_only: Callable[[Path], Path]
) -> None:
    folder = tmp_path / "ro"
    folder.mkdir()
    path, _ = _wal_gpkg(folder)
    read_only(folder)
    assert not os.access(folder, os.W_OK)
    geometries, classes = load_vector_labels(path, "label")
    assert classes == {"a": 1} and len(geometries) == 1
    # Reading it left nothing behind.
    assert sorted(entry.name for entry in folder.iterdir()) == ["labels.gpkg"]


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="a read-only folder only stops users that are not root",
)
def test_unmerged_changes_in_a_read_only_folder_are_named_not_called_corruption(
    tmp_path: Path, read_only: Callable[[Path], Path]
) -> None:
    import shutil

    live = tmp_path / "live"
    live.mkdir()
    path, connection = _wal_gpkg(live, keep_wal=True)
    try:
        folder = tmp_path / "ro"
        folder.mkdir()
        for name in ("labels.gpkg", "labels.gpkg-wal"):
            shutil.copy(live / name, folder / name)
        assert (folder / "labels.gpkg-wal").stat().st_size > 0
    finally:
        assert connection is not None
        connection.close()
    read_only(folder)
    with pytest.raises(ValueError, match=r"read-only folder and has changes that are not merged"):
        load_vector_labels(folder / path.name, "label")
