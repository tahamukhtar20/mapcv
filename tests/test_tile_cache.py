"""The on-disk XYZ tile cache: freshness from HTTP caching headers (RFC 9111), the
file format, and generation against a local tile server that counts its requests."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from mapcv import tile_cache
from mapcv._mapcv_rs import TileIndex, fetch_tiles
from mapcv.cli import app
from mapcv.config import MapcvConfig
from mapcv.pipeline import run_generate
from mapcv.tile_cache import DEFAULT_TTL, TileCache, freshness_lifetime

runner = CliRunner()
NOW = 1_800_000_000.0
NONE: tuple[None, None, None, None] = (None, None, None, None)


def _date(offset: float) -> str:
    return formatdate(NOW + offset, usegmt=True)


# ── Freshness ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("headers", "lifetime"),
    [
        (NONE, DEFAULT_TTL),
        (("public", None, None, None), DEFAULT_TTL),
        (("max-age=3600", None, None, None), 3600.0),
        (("public, Max-Age=60", None, None, None), 60.0),
        (('max-age="90"', None, None, None), 90.0),
        (("max-age=3600", None, None, "600"), 3000.0),
        (("max-age=3600", None, None, "bogus"), 3600.0),
        (("max-age=600", None, None, "600"), None),
        (("max-age=0", None, None, None), None),
        (("max-age=soon", None, None, None), None),
        (("no-store", None, None, None), None),
        (("max-age=3600, no-cache", None, None, None), None),
        (("private, max-age=120", None, None, None), 120.0),
        # max-age wins over Expires.
        (("max-age=10", _date(500), _date(0), None), 10.0),
        ((None, _date(500), _date(0), None), 500.0),
        ((None, _date(500), _date(100), None), 400.0),
        ((None, _date(500), None, None), 500.0),  # no Date: from now
        ((None, _date(500), _date(0), "50"), 450.0),
        ((None, _date(-10), _date(0), None), None),
        ((None, "0", _date(0), None), None),  # invalid Expires: already expired
        ((None, "Tue, 15 Jan 2030 08:00:00", None, None), None),  # no zone: invalid
        ((None, _date(500), "not a date", None), 500.0),
    ],
)
def test_freshness_lifetime(headers: Any, lifetime: float | None) -> None:
    found = freshness_lifetime(headers, NOW)
    if lifetime is None:
        assert found is None
    else:
        assert found == pytest.approx(lifetime)


# ── The cache files ──────────────────────────────────────────────────────────


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


def test_put_get_expiry_and_refusals(tmp_path: Path) -> None:
    clock = Clock()
    cache = TileCache("https://t.example/{z}/{x}/{y}.png?key=SECRET", tmp_path, clock)
    assert cache.get(1, 2, 3) is None
    assert cache.put(1, 2, 3, b"tile", ("max-age=100", None, None, None))
    assert cache.get(1, 2, 3) == b"tile" and cache.get(2, 1, 3) is None
    clock.now += 99
    assert cache.get(1, 2, 3) == b"tile"
    clock.now += 1
    assert cache.get(1, 2, 3) is None  # expired
    assert cache.put(1, 2, 3, b"newer", NONE)  # replaced, default lifetime
    assert cache.get(1, 2, 3) == b"newer"
    clock.now += DEFAULT_TTL
    assert cache.get(1, 2, 3) is None
    assert not cache.put(5, 5, 3, b"x", ("no-store", None, None, None))
    assert cache.get(5, 5, 3) is None and not (cache.folder / "3" / "5").exists()

    # The template, with its key, is never written: the folder is a hash of it.
    paths = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")]
    assert all("SECRET" not in p and "example" not in p for p in paths)
    assert (
        cache.folder.name
        == hashlib.sha256(b"https://t.example/{z}/{x}/{y}.png?key=SECRET").hexdigest()[:24]
    )
    other = TileCache("https://t.example/{z}/{x}/{y}.png?key=OTHER", tmp_path, clock)
    other.put(1, 2, 3, b"other", NONE)
    assert other.get(1, 2, 3) == b"other" and cache.get(1, 2, 3) is None


def test_damaged_files_are_misses(tmp_path: Path) -> None:
    cache = TileCache("t/{z}/{x}/{y}", tmp_path, Clock())
    cache.put(1, 1, 1, b"0123456789", NONE)
    path = cache.folder / "1" / "1" / "1.tile"
    data = path.read_bytes()
    path.write_bytes(data[:-1])  # truncated payload
    assert cache.get(1, 1, 1) is None
    path.write_bytes(data[:10])  # truncated header
    assert cache.get(1, 1, 1) is None
    path.write_bytes(b"x" * len(data))  # not a cache file
    assert cache.get(1, 1, 1) is None
    path.write_bytes(data)
    assert cache.get(1, 1, 1) == b"0123456789"


def test_an_unwritable_cache_warns_once_and_switches_off(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    cache = TileCache("t/{z}/{x}/{y}", blocker, Clock())
    with pytest.warns(UserWarning, match="can't be written") as caught:
        stored = [cache.put(1, 1, 1, b"a", NONE), cache.put(1, 2, 1, b"b", NONE)]
    assert stored == [False, False]
    assert len(caught) == 1 and "imagery.cache: false" in str(caught[0].message)
    assert cache.get(1, 1, 1) is None


def test_usage_and_clear(tmp_path: Path) -> None:
    clock = Clock()
    root = tile_cache.tiles_dir()
    a = TileCache("a/{z}/{x}/{y}", root, clock)
    b = TileCache("b/{z}/{x}/{y}", root, clock)
    a.put(0, 0, 1, b"12345", ("max-age=10", None, None, None))
    a.put(1, 0, 1, b"123", NONE)
    b.put(0, 0, 1, b"1", ("max-age=10", None, None, None))
    (root / "keep.txt").write_text("not a tile")
    (root / "folder.tile").mkdir()  # not a file: never counted or deleted
    (root / "folder.tile" / "inside").write_text("")
    (root / "foreign.tile").write_bytes(b"not a cache file")  # unreadable: expired
    found = tile_cache.usage(now=NOW + 20)
    assert (found.tiles, found.expired) == (4, 3) and found.bytes > 9
    assert tile_cache.usage(now=NOW).expired == 1

    assert tile_cache.clear(expired_only=True, now=NOW + 20) == 3
    assert (root / "folder.tile").is_dir() and not (root / "foreign.tile").exists()
    assert a.get(1, 0, 1) == b"123" and not b.folder.exists()
    assert tile_cache.clear() == 1
    assert tile_cache.usage().tiles == 0 and (root / "keep.txt").exists()
    assert (root / "folder.tile" / "inside").exists()
    assert not a.folder.exists()


def test_cache_dir_per_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(tile_cache.CACHE_ENV, str(tmp_path / "mine"))
    assert tile_cache.cache_dir() == tmp_path / "mine"
    assert tile_cache.tiles_dir() == tmp_path / "mine" / "tiles"
    monkeypatch.delenv(tile_cache.CACHE_ENV)
    monkeypatch.setattr("mapcv.tile_cache.Path.home", lambda: tmp_path)
    monkeypatch.setattr("mapcv.tile_cache.sys.platform", "linux")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert tile_cache.cache_dir() == tmp_path / ".cache" / "mapcv"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert tile_cache.cache_dir() == tmp_path / "xdg" / "mapcv"
    monkeypatch.setattr("mapcv.tile_cache.sys.platform", "darwin")
    assert tile_cache.cache_dir() == tmp_path / "Library" / "Caches" / "mapcv"
    monkeypatch.setattr("mapcv.tile_cache.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert tile_cache.cache_dir() == tmp_path / "local" / "mapcv" / "Cache"
    monkeypatch.delenv("LOCALAPPDATA")
    assert tile_cache.cache_dir() == tmp_path / "AppData" / "Local" / "mapcv" / "Cache"


# ── Against a tile server ────────────────────────────────────────────────────


def _png(x: int, y: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (256, 256), ((x * 37) % 256, (y * 53) % 256, 120)).save(buffer, "PNG")
    return buffer.getvalue()


class TileServer:
    """Serves ``/{z}/{x}/{y}.png`` with chosen caching headers; counts requests."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.headers: dict[str, str] = {}
        self.failing: set[tuple[int, int]] = set()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                owner.requests.append(self.path)
                _z, x, y = (int(part) for part in self.path.strip("/").split(".")[0].split("/"))
                if (x, y) in owner.failing:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = _png(x, y)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                for name, value in owner.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.template = f"http://127.0.0.1:{self.server.server_address[1]}/{{z}}/{{x}}/{{y}}.png"


@pytest.fixture
def server() -> Iterator[TileServer]:
    tiles = TileServer()
    yield tiles
    tiles.server.shutdown()
    tiles.server.server_close()


def test_fetch_tiles_returns_caching_headers(server: TileServer) -> None:
    server.headers = {"Cache-Control": "max-age=60", "Age": "5"}
    server.failing = {(2, 1)}
    tiles = [TileIndex(1, 1, 2), TileIndex(2, 1, 2)]
    results, failed, _ = fetch_tiles(tiles, server.template, policy="ignore", cache_headers=True)
    by_tile = {(t.x, t.y): (payload, headers) for t, payload, headers in results}
    assert failed == 1
    payload, headers = by_tile[(1, 1)]
    assert payload == _png(1, 1) and headers is not None
    cache_control, expires, date, age = headers
    assert (cache_control, expires, age) == ("max-age=60", None, "5") and date
    assert by_tile[(2, 1)][1] is None  # a black fill has no headers
    plain, _, _ = fetch_tiles(tiles[:1], server.template)
    assert len(plain[0]) == 2  # without cache_headers: (tile, bytes) as before


def _config(server: TileServer, staging: Path, **imagery: Any) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": {"west": 4.9, "south": 52.30, "east": 4.915, "north": 52.31},
            "imagery": {
                "type": "xyz",
                "zoom": 16,
                "url_template": server.template,
                "max_connections": 4,
                **imagery,
            },
            "sampler": {"patch_size": 256, "edge_strategy": "drop", "max_empty_ratio": 1.0},
            "writer": {"staging_dir": str(staging)},
        }
    )


def _files(staging: Path) -> dict[str, bytes]:
    return {
        p.relative_to(staging).as_posix(): p.read_bytes()
        for p in sorted(staging.rglob("*"))
        if p.is_file()
    }


def test_a_second_run_reads_every_tile_from_the_cache(server: TileServer, tmp_path: Path) -> None:
    first = run_generate(_config(server, tmp_path / "a"))
    downloaded = len(server.requests)
    assert downloaded > 4 and first.tiles_requested == downloaded and first.tiles_cached == 0
    assert tile_cache.usage().tiles == downloaded

    second = run_generate(_config(server, tmp_path / "b"))
    assert len(server.requests) == downloaded  # nothing downloaded again
    assert (second.tiles_requested, second.tiles_cached) == (0, downloaded)
    assert _files(tmp_path / "a") == _files(tmp_path / "b")

    result = runner.invoke(app, ["cache"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert f"{downloaded} tiles" in result.output and "expired" not in result.output


def test_headers_and_settings_that_keep_tiles_out(server: TileServer, tmp_path: Path) -> None:
    server.headers = {"Cache-Control": "no-store"}
    run_generate(_config(server, tmp_path / "a"))
    first = len(server.requests)
    run_generate(_config(server, tmp_path / "b"))
    assert len(server.requests) == 2 * first and tile_cache.usage().tiles == 0

    server.headers = {}
    run_generate(_config(server, tmp_path / "c", cache=False))
    assert len(server.requests) == 3 * first and tile_cache.usage().tiles == 0

    # Failed tiles are filled with black and never cached: only they are asked for again.
    server.failing = {next(iter(_requested_tiles(server)))}
    run_generate(_config(server, tmp_path / "d", policy="ignore"))
    assert tile_cache.usage().tiles == first - 1
    before = len(server.requests)
    again = run_generate(_config(server, tmp_path / "e", policy="ignore"))
    assert len(server.requests) == before + 1
    assert (again.tiles_requested, again.tiles_cached, again.tiles_failed) == (1, first - 1, 1)


def _requested_tiles(server: TileServer) -> list[tuple[int, int]]:
    tiles = []
    for path in server.requests:
        _, x, y = path.strip("/").split(".")[0].split("/")
        tiles.append((int(x), int(y)))
    return tiles


def test_expired_tiles_are_downloaded_again(
    server: TileServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.headers = {"Cache-Control": "max-age=100"}
    run_generate(_config(server, tmp_path / "a"))
    first = len(server.requests)
    import time

    real = time.time
    monkeypatch.setattr("mapcv.tile_cache.time.time", lambda: real() + 101)
    run_generate(_config(server, tmp_path / "b"))
    assert len(server.requests) == 2 * first


# ── CLI and MCP ──────────────────────────────────────────────────────────────


def test_cache_cli(tmp_path: Path) -> None:
    env = {"COLUMNS": "200"}
    cache = TileCache("t/{z}/{x}/{y}", tile_cache.tiles_dir())
    cache.put(1, 1, 1, b"a" * 1000, NONE)
    cache.put(1, 2, 1, b"b", ("max-age=1", None, None, "5"))  # stale on arrival: not kept
    result = runner.invoke(app, ["cache"], env=env)
    assert result.exit_code == 0 and "1 tile, 1.0 KB" in result.output
    assert str(tile_cache.tiles_dir()) in result.output.replace("\n", "")
    result = runner.invoke(app, ["cache", "--expired"], env=env)
    assert result.exit_code == 2 and "only applies with --clear" in result.output
    result = runner.invoke(app, ["cache", "--clear", "--expired"], env=env)
    assert result.exit_code == 0 and "Deleted 0 expired cached tiles" in result.output
    result = runner.invoke(app, ["cache", "--clear"], env=env)
    assert result.exit_code == 0 and "Deleted 1 cached tile from" in result.output
    assert tile_cache.usage().tiles == 0


def test_generate_summary_counts_cached_tiles(server: TileServer, tmp_path: Path) -> None:
    import yaml

    data = _config(server, tmp_path / "a").model_dump(mode="json", exclude_none=True)
    for name in ("a", "b"):
        data["writer"]["staging_dir"] = str(tmp_path / name)
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(data))
    env = {"COLUMNS": "200"}
    first = runner.invoke(app, ["generate", str(tmp_path / "a.yaml"), "--yes"], env=env)
    assert first.exit_code == 0, first.output
    assert "from the cache" not in first.output
    second = runner.invoke(app, ["generate", str(tmp_path / "b.yaml"), "--yes"], env=env)
    assert second.exit_code == 0, second.output
    assert f"0 fetched · {len(server.requests)} from the cache · 0 failed" in second.output


def test_mcp_generate_never_uses_the_cache(server: TileServer, tmp_path: Path) -> None:
    import yaml

    from mapcv.agent_tools import Sandbox, ToolState, prepare_generate

    data = _config(server, tmp_path / "a").model_dump(mode="json", exclude_none=True)
    data["writer"]["staging_dir"] = "out"
    (tmp_path / "one.yaml").write_text(yaml.safe_dump(data))
    data["imagery"] = [{**data["imagery"], "name": "x"}, {**data["imagery"], "name": "y"}]
    (tmp_path / "two.yaml").write_text(yaml.safe_dump(data))
    state = ToolState(Sandbox(tmp_path, allow_write=True))
    one = prepare_generate(state, "one.yaml").config
    assert one.primary_imagery.cache is False  # type: ignore[union-attr]
    two = prepare_generate(state, "two.yaml").config
    assert [source.cache for source in two.sources] == [False, False]  # type: ignore[union-attr]

    from mapcv.agent_tools import _without_tile_cache

    geotiff = MapcvConfig.model_validate(
        {**data, "imagery": [{"type": "geotiff", "name": "g", "path": "a.tif"}, data["imagery"][0]]}
    )
    kept, xyz = _without_tile_cache(geotiff).sources
    assert kept is geotiff.sources[0] and xyz.cache is False  # type: ignore[union-attr]
