"""Reads and writes stay where they belong: links and paths in a dataset folder, STAC
catalogs, the MCP server's root, and credentials in what mapcv records or shows."""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
import urllib.error
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("rasterio", reason="the GeoTIFF is written with rasterio")
from test_geotiff_imagery import config_for, make_raster, write_labels

from mapcv import MapcvDataset, generate, split, stac
from mapcv._confine import (
    LinkEscapeError,
    check_folder_links,
    check_inside,
    first_outside_path,
    is_dataset_path,
)
from mapcv.cli import app
from mapcv.config import LabelsConfig, MapcvConfig, OsmLabelsSource
from mapcv.manifest import Manifest, ManifestMismatchError
from mapcv.pipeline import run_generate, run_split
from mapcv.verify import verify_dataset

runner = CliRunner()


def _config(folder: Path, staging: str = "dataset") -> Path:
    """A GeoTIFF + labels config file in ``folder`` (no network)."""
    folder.mkdir(parents=True, exist_ok=True)
    raster = make_raster(folder, width=320, height=256, count=3)
    region = raster.region()
    labels = write_labels(folder, region)
    config = config_for(folder, {"path": str(raster.path)}, region, labels=labels, staging=staging)
    data = config.model_dump(mode="json", exclude_unset=True)
    data["split"] = {"strategy": "random", "test_ratio": 0.2, "val_ratio": 0.1}
    path = folder / "mapcv.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _dataset(folder: Path) -> Path:
    config = MapcvConfig.from_yaml(_config(folder))
    run_generate(config)
    return Path(config.writer.staging_dir)


# ── Links in a dataset folder (D-1) ──────────────────────────────────────────


def test_generate_does_not_write_through_a_link_out_of_the_staging_dir(tmp_path: Path) -> None:
    config = _config(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("precious\n")
    staging = tmp_path / "dataset"
    (staging / "Images").mkdir(parents=True)
    (staging / "Images" / "patch_0000000.png").symlink_to(outside / "a.txt")
    (staging / "splits").mkdir()
    (staging / "splits" / "train.txt").symlink_to(outside / "c.txt")

    result = runner.invoke(app, ["generate", str(config), "--yes"])

    assert result.exit_code == 1
    assert "Unsafe link" in result.output and "outside" in result.output
    assert (outside / "a.txt").read_text() == "precious\n"
    assert not (outside / "c.txt").exists()


@pytest.mark.parametrize(
    "command",
    [
        ["stats"],
        ["card", "--force"],
        ["split", "--strategy", "random"],
        ["verify", "--write-checksums"],
        ["export", "-f", "terratorch"],
    ],
)
def test_dataset_commands_refuse_a_link_out_of_the_dataset(
    tmp_path: Path, command: list[str]
) -> None:
    staging = _dataset(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("precious\n")
    for name in ("stats.json", "README.md", "terratorch.yaml", "SHA256SUMS"):
        (staging / name).unlink(missing_ok=True)
        (staging / name).symlink_to(outside)

    result = runner.invoke(app, [command[0], str(staging), *command[1:]])

    assert result.exit_code == 1, result.output
    assert "Unsafe link" in result.output
    assert outside.read_text() == "precious\n"


def test_links_inside_the_dataset_are_fine(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    (staging / "manifest_alias.json").symlink_to(staging / "manifest.json")
    result = runner.invoke(app, ["stats", str(staging)])
    assert result.exit_code == 0, result.output


def test_check_folder_links(tmp_path: Path) -> None:
    folder = tmp_path / "ds"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.txt").write_text("a")
    (folder / "inside").symlink_to(folder / "sub" / "a.txt")
    check_folder_links(folder, "ds")
    check_folder_links(tmp_path / "missing", "ds")  # nothing to check
    (tmp_path / "other.txt").write_text("b")
    os.link(tmp_path / "other.txt", folder / "sub" / "hard.txt")
    check_folder_links(folder, "ds")  # hard links count only when asked
    with pytest.raises(LinkEscapeError, match="sub/hard.txt is a hard link"):
        check_folder_links(folder, "ds", hard_links=True)
    (folder / "sub" / "hard.txt").unlink()
    (folder / "sub" / "out").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(LinkEscapeError, match="sub/out is a link to"):
        check_folder_links(folder, "ds")


def test_check_inside(tmp_path: Path) -> None:
    folder = tmp_path / "ds"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.txt").write_text("a")
    (folder / "alias.txt").symlink_to(folder / "sub" / "a.txt")
    (tmp_path / "other.txt").write_text("b")
    (folder / "out.txt").symlink_to(tmp_path / "other.txt")
    (folder / "outdir").symlink_to(tmp_path, target_is_directory=True)
    check_inside(folder, folder / "sub" / "a.txt", "ds")
    check_inside(folder, folder / "alias.txt", "ds")
    check_inside(folder, folder / "missing.txt", "ds")  # reading it fails on its own
    with pytest.raises(LinkEscapeError, match="out.txt leads through a link to"):
        check_inside(folder, folder / "out.txt", "ds")
    with pytest.raises(LinkEscapeError, match="outdir/other.txt leads through a link to"):
        check_inside(folder, folder / "outdir" / "other.txt", "ds")
    with pytest.raises(LinkEscapeError):
        check_inside(folder, folder / "sub" / ".." / ".." / "other.txt", "ds")
    with pytest.raises(LinkEscapeError, match="other.txt leads through a link"):
        check_inside(folder, tmp_path / "other.txt", "ds")  # not below the folder at all


def _link_out(staging: Path, rel: str, tmp_path: Path) -> Path:
    """Replace ``rel`` in the dataset with a link to a copy of it outside; the copy."""
    outside = tmp_path / ("outside_" + rel.replace("/", "_"))
    path = staging / rel
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    return outside


@pytest.mark.parametrize("rel", ["manifest.json", "splits/train.txt"])
def test_info_does_not_read_through_a_link_out_of_the_dataset(tmp_path: Path, rel: str) -> None:
    staging = _dataset(tmp_path)
    _link_out(staging, rel, tmp_path)

    result = runner.invoke(app, ["info", str(staging)])

    assert result.exit_code == 1, result.output
    assert "Unsafe link" in result.output and "outside" in result.output


def test_info_refuses_a_splits_folder_linked_out_of_the_dataset(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    (staging / "splits").rename(tmp_path / "splits")
    (staging / "splits").symlink_to(tmp_path / "splits", target_is_directory=True)

    result = runner.invoke(app, ["info", str(staging)])

    assert result.exit_code == 1, result.output
    assert "Unsafe link" in result.output


def test_info_reads_a_dataset_with_links_inside_it(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    (staging / "splits" / "train.txt").rename(staging / "train_list.txt")
    (staging / "splits" / "train.txt").symlink_to(staging / "train_list.txt")
    result = runner.invoke(app, ["info", str(staging)])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("rel", ["manifest.json", "splits/train.txt"])
def test_mapcv_dataset_refuses_a_linked_manifest_or_split_list(tmp_path: Path, rel: str) -> None:
    staging = _dataset(tmp_path)
    _link_out(staging, rel, tmp_path)
    with pytest.raises(LinkEscapeError, match="outside"):
        MapcvDataset(staging, "train", as_tensors=False)


def test_mapcv_dataset_refuses_a_linked_patch_file(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    dataset = MapcvDataset(staging, "all", as_tensors=False)
    image = dataset.entries[0]["files"]["image"]
    _link_out(staging, image, tmp_path)
    with pytest.raises(LinkEscapeError, match=f"{image} leads through a link"):
        dataset[0]
    mask = dataset.entries[1]["files"]["mask"]
    _link_out(staging, mask, tmp_path)
    with pytest.raises(LinkEscapeError, match=f"{mask} leads through a link"):
        dataset[1]


def test_mapcv_dataset_reads_links_inside_the_dataset(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    dataset = MapcvDataset(staging, "all", as_tensors=False)
    before = dataset[0]
    path = staging / dataset.entries[0]["files"]["image"]
    path.rename(staging / "moved.png")
    path.symlink_to(staging / "moved.png")
    after = MapcvDataset(staging, "all", as_tensors=False)[0]
    assert (after["image"] == before["image"]).all()


@pytest.mark.parametrize("entry_point", ["run_generate", "generate"])
def test_library_generate_refuses_a_link_out_of_the_staging_dir(
    tmp_path: Path, entry_point: str
) -> None:
    config = MapcvConfig.from_yaml(_config(tmp_path))
    outside = tmp_path / "outside.txt"
    outside.write_text("precious\n")
    staging = Path(config.writer.staging_dir)
    (staging / "Images").mkdir(parents=True)
    (staging / "Images" / "patch_0000000.png").symlink_to(outside)

    with pytest.raises(LinkEscapeError, match="writer.staging_dir"):
        run_generate(config) if entry_point == "run_generate" else generate(config)
    assert outside.read_text() == "precious\n"


@pytest.mark.parametrize("entry_point", ["run_split", "split"])
def test_library_split_refuses_a_link_out_of_the_dataset(tmp_path: Path, entry_point: str) -> None:
    staging = _dataset(tmp_path)
    outside = _link_out(staging, "splits/train.txt", tmp_path)
    before = outside.read_bytes()

    with pytest.raises(LinkEscapeError, match="splits/train.txt is a link"):
        run_split(staging) if entry_point == "run_split" else split(staging)
    assert outside.read_bytes() == before


def _library_call(name: str, staging: Path, out: Path) -> object:
    """Call the library function behind ``mapcv stats``, ``card``, ``verify`` or ``export``."""
    from mapcv import card, export, shards, stats, verify

    calls: dict[str, Callable[[], object]] = {
        "dataset_stats": lambda: stats.dataset_stats(staging),
        "write_stats": lambda: stats.write_stats(staging),
        "card_text": lambda: card.card_text(staging),
        "write_card": lambda: card.write_card(staging, overwrite=True),
        "verify_dataset": lambda: verify.verify_dataset(staging, deep=True),
        "write_checksums": lambda: verify.write_checksums(staging),
        "terratorch_config": lambda: export.terratorch_config(staging),
        "export_terratorch": lambda: export.export_terratorch(staging),
        "export_hf_parquet": lambda: export.export_hf_parquet(staging, out),
        "export_webdataset": lambda: shards.export_webdataset(staging, out),
        "export_zarr": lambda: shards.export_zarr(staging, out),
    }
    return calls[name]()


_LIBRARY_CALLS = [
    "dataset_stats",
    "write_stats",
    "card_text",
    "write_card",
    "verify_dataset",
    "write_checksums",
    "terratorch_config",
    "export_terratorch",
    "export_hf_parquet",
    "export_webdataset",
    "export_zarr",
]
_NEEDS = {"export_hf_parquet": "pyarrow", "export_zarr": "zarr"}


@pytest.mark.parametrize("name", _LIBRARY_CALLS)
def test_library_dataset_functions_refuse_a_link_out_of_the_dataset(
    tmp_path: Path, name: str
) -> None:
    if name in _NEEDS:
        pytest.importorskip(_NEEDS[name])
    staging = _dataset(tmp_path)
    image = Manifest.load(staging / "manifest.json").patches[0]["files"]["image"]
    outside = _link_out(staging, image, tmp_path)
    before = outside.read_bytes()
    (tmp_path / "outside.txt").write_text("precious\n")
    for rel in ("stats.json", "README.md", "terratorch.yaml", "SHA256SUMS"):
        (staging / rel).unlink(missing_ok=True)
        (staging / rel).symlink_to(tmp_path / "outside.txt")
    out = tmp_path / "export"

    with pytest.raises(LinkEscapeError, match="the dataset folder: .* is a link to"):
        _library_call(name, staging, out)
    assert outside.read_bytes() == before
    assert (tmp_path / "outside.txt").read_text() == "precious\n"
    assert not out.exists()


@pytest.mark.parametrize("name", _LIBRARY_CALLS)
def test_library_dataset_functions_read_links_inside_the_dataset(tmp_path: Path, name: str) -> None:
    if name in _NEEDS:
        pytest.importorskip(_NEEDS[name])
    staging = _dataset(tmp_path)
    image = staging / Manifest.load(staging / "manifest.json").patches[0]["files"]["image"]
    image.rename(staging / "moved.png")
    image.symlink_to(staging / "moved.png")
    _library_call(name, staging, tmp_path / "export")


def test_mapcv_dataset_refuses_a_link_out_of_a_zarr_store(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    from mapcv.shards import export_zarr

    store = export_zarr(_dataset(tmp_path), tmp_path / "d.zarr")
    MapcvDataset(store, "all", as_tensors=False)[0]
    chunk = min(path for path in (store / "images").rglob("*") if path.name[0].isdigit())
    rel = chunk.relative_to(store).as_posix()
    _link_out(store, rel, tmp_path)

    with pytest.raises(LinkEscapeError, match=f"{rel} is a link to"):
        MapcvDataset(store, "all", as_tensors=False)


# ── Paths in a manifest and in SHA256SUMS (D-3) ──────────────────────────────


@pytest.mark.parametrize(
    "rel", ["Images/patch_0000000.png", "Images/s2/patch_0.npy", "Masks/x.png", "A/p.tif"]
)
def test_dataset_paths(rel: str) -> None:
    assert is_dataset_path(rel)


@pytest.mark.parametrize(
    "rel", ["", "/etc/passwd", "../x.png", "Images/../../x", "Images//x", "./x", "C:/x", "a\\b"]
)
def test_paths_that_leave_the_dataset(rel: str) -> None:
    assert not is_dataset_path(rel)
    assert first_outside_path(["Images/a.png", rel, "Masks/b.png"]) == rel


def test_the_fast_check_agrees_with_the_path_rule() -> None:
    """``first_outside_path`` scans a whole manifest at once; it must refuse exactly what
    ``is_dataset_path`` refuses."""
    alphabet = ["a", "/", ".", "..", "\\", ":", "\n", "\x00"]
    for length in range(5):
        for parts in itertools.product(alphabet, repeat=length):
            rel = "".join(parts)
            found = first_outside_path(["Images/x.png", rel])
            assert (found is None) == is_dataset_path(rel), repr(rel)
    assert first_outside_path([]) is None


@pytest.mark.parametrize("bad", ["{outside}", "../{name}"])
def test_a_manifest_path_outside_the_dataset_is_refused(tmp_path: Path, bad: str) -> None:
    staging = _dataset(tmp_path / "work")
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-FILE-CONTENT\n")
    path = bad.format(outside=outside, name=outside.name)
    manifest = staging / "manifest.json"
    first = Manifest.load(manifest).patches[0]["files"]["image"]
    manifest.write_text(manifest.read_text().replace(f'"{first}"', json.dumps(path), 1))

    with pytest.raises(ManifestMismatchError, match="not a path inside the dataset folder"):
        Manifest.load(manifest)
    report = verify_dataset(staging)
    assert not report.ok and "not a path inside the dataset folder" in report.problems[0]
    exported = runner.invoke(
        app, ["export", str(staging), "-f", "webdataset", "-o", str(tmp_path / "wds")]
    )
    assert exported.exit_code == 1
    shards = list((tmp_path / "wds").glob("*.tar"))
    assert not any(b"OUTSIDE-FILE-CONTENT" in shard.read_bytes() for shard in shards)


def test_a_checksum_path_outside_the_dataset_is_a_problem(tmp_path: Path) -> None:
    staging = _dataset(tmp_path)
    (staging / "SHA256SUMS").write_text("0000  /etc/hostname\n0000  ../mapcv.yaml\n")
    report = verify_dataset(staging)
    assert report.checked_hashes == 0
    assert report.problems == [
        "SHA256SUMS lists '/etc/hostname', which is not a path inside the dataset folder",
        "SHA256SUMS lists '../mapcv.yaml', which is not a path inside the dataset folder",
    ]


# ── STAC catalogs (E-13, E-14) ───────────────────────────────────────────────

_WORLD = {
    "type": "Polygon",
    "coordinates": [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]],
}


def _item(href: str) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": "item",
        "geometry": _WORLD,
        "properties": {"datetime": "2024-06-01T00:00:00Z", "eo:cloud_cover": 1.0},
        "assets": {"red": {"href": href}, "green": {"href": href}, "blue": {"href": href}},
    }


@pytest.fixture
def catalog() -> Iterator[dict[str, Any]]:
    """A STAC API answering every search with ``state["page"]``."""
    state: dict[str, Any] = {"page": {"features": [], "links": []}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps(state["page"]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def _search(url: str) -> Any:
    from mapcv.config import StacSearchBase

    return StacSearchBase.model_validate({"catalog": url, "datetime": "2024-06-01"})


@pytest.mark.parametrize(
    "href", ["file:///etc/passwd", "http://169.254.169.254/latest", "https://elsewhere.example/p"]
)
def test_a_next_link_stays_on_the_catalog_host(catalog: dict[str, Any], href: str) -> None:
    catalog["page"] = {"features": [], "links": [{"rel": "next", "href": href, "method": "GET"}]}
    with pytest.raises(RuntimeError, match="next links on the catalog's own scheme and host"):
        list(stac.search_items(_search(catalog["url"]), (0, 0, 1, 1)))


def test_a_relative_next_link_on_the_catalog_host_is_followed(
    catalog: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog["page"] = {
        "features": [],
        "links": [{"rel": "next", "href": "/search?page=2", "method": "POST"}],
    }
    pages = 0
    original = stac._post

    def counting(url: str, body: dict[str, Any]) -> dict[str, Any]:
        nonlocal pages
        pages += 1
        if pages > 1:
            assert url == catalog["url"] + "/search?page=2"
            return {"features": [], "links": []}
        return original(url, body)

    monkeypatch.setattr(stac, "_post", counting)
    assert list(stac.search_items(_search(catalog["url"]), (0, 0, 1, 1))) == []
    assert pages == 2


def test_local_assets_go_through_the_active_check(catalog: dict[str, Any], tmp_path: Path) -> None:
    catalog["page"] = {"features": [_item(str(tmp_path / "outside.tif"))], "links": []}
    search = _search(catalog["url"])
    assert stac.find_item(search, (0, 0, 1, 1))[0]["id"] == "item"  # no check: allowed
    checked: list[Path] = []

    def refuse(path: Path) -> None:
        checked.append(path)
        raise PermissionError(f"{path} refused")

    with (
        stac.local_paths_checked(refuse),
        pytest.raises(PermissionError, match="outside.tif refused"),
    ):
        stac.find_item(search, (0, 0, 1, 1))
    assert checked == [tmp_path / "outside.tif"]
    catalog["page"] = {"features": [_item("https://example.org/a.tif")], "links": []}
    with stac.local_paths_checked(refuse):
        stac.find_item(search, (0, 0, 1, 1))  # remote assets are not local files


class _Endless:
    """A response body that never ends (``delay`` seconds per read)."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def read1(self, size: int) -> bytes:
        time.sleep(self.delay)
        return b" " * min(size, 1 << 16)


def test_a_stac_page_has_a_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stac, "_MAX_PAGE_BYTES", 1 << 20)
    with pytest.raises(RuntimeError, match="sent more than 1 MiB for one page"):
        stac._read_json(_Endless(), "http://127.0.0.1:1/search")


def test_a_stac_page_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stac, "_DEADLINE_S", 0.3)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="took longer than 0.3 s"):
        stac._read_json(_Endless(delay=0.1), "http://127.0.0.1:1/search")
    assert time.monotonic() - started < 2


def test_a_stac_page_must_be_a_json_object() -> None:
    class Once:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def read1(self, size: int) -> bytes:
            body, self.body = self.body, b""
            return body

    with pytest.raises(RuntimeError, match="something other than JSON"):
        stac._read_json(Once(b"<html>"), "http://127.0.0.1:1/search")
    with pytest.raises(RuntimeError, match="not a STAC search page"):
        stac._read_json(Once(b"[1, 2]"), "http://127.0.0.1:1/search")


# ── The Overpass URL's key (M-7) ─────────────────────────────────────────────


def _osm(url: str) -> OsmLabelsSource:
    return OsmLabelsSource.model_validate(
        {"overpass_url": url, "classes": [{"name": "building", "tags": {"building": "*"}}]}
    )


def test_the_overpass_key_is_not_recorded(tmp_path: Path) -> None:
    from mapcv.targets.segmentation import label_settings

    url = "http://127.0.0.1:8705/api/interpreter?key=SUPERSECRET123&format=json"
    labels = LabelsConfig.model_validate({"osm": _osm(url).model_dump()})
    settings = label_settings(labels, set())
    assert settings["osm"]["overpass_url"] == (
        "http://127.0.0.1:8705/api/interpreter?key=***&format=***"
    )
    assert "SUPERSECRET123" not in json.dumps(settings)


def test_the_overpass_key_is_not_shown_in_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from mapcv import osm

    def fail(*args: Any, **kwargs: Any) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("mapcv.osm.urlopen", fail)
    source = _osm("http://127.0.0.1:1/api/interpreter?key=SUPERSECRET123")
    with pytest.raises(RuntimeError) as excinfo:
        osm._fetch(source, "[out:json];")
    assert "SUPERSECRET123" not in str(excinfo.value)
    assert "http://127.0.0.1:1/api/interpreter failed" in str(excinfo.value)
