"""``mapcv doctor``: the checks, the table and JSON output, the exit code, and that no secret
ever reaches the output."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mapcv import doctor
from mapcv.cli import app
from mapcv.doctor import Check, Endpoint, TerminalInfo

runner = CliRunner()
SECTIONS = ["mapcv", "Core", "Extras", "Tile cache", "Terminal", "Network"]
SENTINEL = "SENTINEL-4f9c1e7a-do-not-print"
PROXY_VARIABLES = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY")


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probes of 127.0.0.1 must go direct, whatever the machine running the tests sets."""
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def _json(args: list[str]) -> tuple[int, dict[str, Any]]:
    result = runner.invoke(app, ["doctor", "--json", *args])
    return result.exit_code, json.loads(result.output)


def _by_name(document: dict[str, Any], name: str) -> dict[str, Any]:
    found = [check for check in document["checks"] if check["name"] == name]
    assert len(found) == 1, f"{name}: {[c['name'] for c in document['checks']]}"
    result: dict[str, Any] = found[0]
    return result


def _refused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


@contextmanager
def _silent_server() -> Iterator[int]:
    """A server that accepts connections and never answers."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        sock.settimeout(0.1)
        while not stop.is_set():
            try:
                held.append(sock.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield int(sock.getsockname()[1])
    finally:
        stop.set()
        thread.join()
        for connection in held:
            connection.close()
        sock.close()


# --- the command -----------------------------------------------------------------------


def test_an_offline_run_shows_every_section_and_exits_zero() -> None:
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 0, result.output
    for section in SECTIONS:
        assert re.search(rf"^{section}$", result.output, re.MULTILINE), section
    assert "skipped (--offline)" in result.output
    assert "\x1b" not in result.output  # piped: no colour codes


def test_the_table_fits_80_columns() -> None:
    result = runner.invoke(app, ["doctor", "--offline"])
    assert max(len(line) for line in result.output.splitlines()) <= 80


def test_json_has_the_documented_keys() -> None:
    code, document = _json(["--offline"])
    assert code == 0
    assert set(document) == {"ok", "offline", "checks"}
    assert document["ok"] is True and document["offline"] is True
    seen = []
    for check in document["checks"]:
        assert set(check) == {"section", "name", "status", "detail", "hint"}
        assert check["status"] in ("ok", "warn", "fail", "info")
        assert isinstance(check["detail"], str) and check["detail"]
        assert check["hint"] is None or isinstance(check["hint"], str)
        if check["section"] not in seen:
            seen.append(check["section"])
    assert seen == SECTIONS
    assert _by_name(document, "Rust extension")["status"] == "ok"


def test_json_is_plain_text_with_unix_line_endings() -> None:
    result = runner.invoke(app, ["doctor", "--json", "--offline"])
    assert result.output.endswith("}\n") and "\r" not in result.output
    assert "\x1b" not in result.output and result.output.isascii()


def test_doctor_is_listed_among_the_utilities() -> None:
    # Typer forces colour on GitHub Actions, so ask for none and strip what remains.
    env = {"COLUMNS": "100", "NO_COLOR": "1", "GITHUB_ACTIONS": "", "FORCE_COLOR": ""}
    result = runner.invoke(app, ["--help"], env=env)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert re.search(r"3\. Utilities.*\n(?:.*\n)*?.*\bdoctor\b", plain), plain
    assert "mapcv doctor" in runner.invoke(app, ["doctor", "--help"], env=env).output


def test_the_core_dependency_list_matches_pyproject() -> None:
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert block
    declared = {
        re.match(r"[A-Za-z0-9._-]+", line.strip().strip('",')).group().lower().replace("_", "-")  # type: ignore[union-attr]
        for line in block.group(1).splitlines()
        if line.strip()
    }
    assert {n.lower().replace("_", "-") for n in doctor._FALLBACK_CORE_DEPENDENCIES} == declared
    # and the live metadata agrees, so the report lists what is really required
    assert {n.lower().replace("_", "-") for n in doctor._core_dependencies()} == declared


# --- the Rust extension ------------------------------------------------------------------


def test_a_mismatched_extension_fails_with_a_reinstall_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mapcv._mapcv_rs.__version__", "0.0.1", raising=False)
    code, document = _json(["--offline"])
    check = _by_name(document, "Rust extension")
    assert code == 1 and document["ok"] is False
    assert check["status"] == "fail" and "0.0.1" in check["detail"]
    assert "pip install --force-reinstall" in check["hint"]


def test_an_extension_that_cannot_be_imported_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> Any:
        if name == "mapcv._mapcv_rs":
            raise ImportError("undefined symbol: PyInit__mapcv_rs")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake)
    code, document = _json(["--offline"])
    check = _by_name(document, "Rust extension")
    assert code == 1 and check["status"] == "fail"
    assert "undefined symbol" in check["detail"] and "reinstall" in check["hint"].lower()


def test_an_extension_without_a_version_still_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr("mapcv._mapcv_rs.__version__", raising=False)
    check = next(c for c in doctor.core_checks() if c.name == "Rust extension")
    assert check.status == "ok"


def test_a_missing_core_dependency_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.metadata.version

    def fake(name: str) -> str:
        if name == "pyshp":
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", fake)
    code, document = _json(["--offline"])
    assert code == 1 and _by_name(document, "pyshp")["status"] == "fail"


# --- extras ----------------------------------------------------------------------------


def _hide_modules(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    real = importlib.util.find_spec

    def fake(name: str, package: str | None = None) -> Any:
        return None if name in names else real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_a_missing_extra_is_info_with_its_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_modules(monkeypatch, "pyarrow", "mcp", "ee", "zarr", "xarray_eopf")
    code, document = _json(["--offline"])
    assert code == 0
    expected = {
        "parquet / export": 'pip install "mapcv[parquet]"',
        "mcp": 'pip install "mapcv[mcp]"',
        "gee": 'pip install "mapcv[gee]"',
    }
    for name, command in expected.items():
        check = _by_name(document, name)
        assert check["status"] == "info", name
        assert "not installed" in check["detail"] and command in check["hint"]
    if sys.version_info[:2] < (3, 14):
        assert 'pip install "mapcv[zarr]"' in _by_name(document, "zarr")["hint"]
    # no credentials row for an extra that is not installed
    assert not [c for c in document["checks"] if c["name"] == "Earth Engine credentials"]


def test_zarr_on_python_3_14_points_to_the_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_modules(monkeypatch, "zarr", "xarray_eopf")
    monkeypatch.setattr(sys, "version_info", (3, 14, 0, "final", 0))
    check = doctor.extras_checks()[0]
    assert check.status == "info" and "3.14" in check.detail
    assert check.hint is not None and "issues/212" in check.hint


def test_a_half_installed_zarr_extra_is_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_modules(monkeypatch, "xarray_eopf")
    if sys.version_info[:2] >= (3, 14) or importlib.util.find_spec("zarr") is None:
        pytest.skip("needs zarr installed on Python < 3.14")
    assert doctor.extras_checks()[0].status == "warn"


def test_mcp_1_is_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("mcp") is None:
        pytest.skip("needs the mcp extra")
    real = doctor._dist_version
    monkeypatch.setattr(
        doctor, "_dist_version", lambda dist: "1.9.0" if dist == "mcp" else real(dist)
    )
    check = next(c for c in doctor.extras_checks() if c.name == "mcp")
    assert check.status == "warn" and "2.x" in check.detail


# --- the tile cache --------------------------------------------------------------------


def test_an_unwritable_cache_fails_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("MAPCV_CACHE_DIR", str(blocker))
    result = runner.invoke(app, ["doctor", "--offline"])
    assert result.exit_code == 1
    code, document = _json(["--offline"])
    folder = _by_name(document, "Folder")
    assert code == 1 and document["ok"] is False and folder["status"] == "fail"
    assert "MAPCV_CACHE_DIR" in folder["hint"] and "can't be written" in folder["detail"]
    assert "FAIL" in result.output and "1 check(s) failed" in result.output
    assert blocker.read_text(encoding="utf-8") == "x"


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)
def test_a_read_only_cache_folder_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    monkeypatch.setenv("MAPCV_CACHE_DIR", str(locked))
    try:
        code, document = _json(["--offline"])
    finally:
        locked.chmod(0o700)
    assert code == 1 and _by_name(document, "Folder")["status"] == "fail"


def test_a_writable_cache_is_left_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MAPCV_CACHE_DIR", str(cache))
    code, document = _json(["--offline"])
    assert code == 0 and _by_name(document, "Folder")["status"] == "ok"
    assert "MAPCV_CACHE_DIR" in _by_name(document, "Folder")["detail"]
    assert list(cache.rglob("*")) == []  # the probe file is gone, nothing else was made


def test_a_cache_folder_that_does_not_exist_yet_is_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "later" / "cache"
    monkeypatch.setenv("MAPCV_CACHE_DIR", str(cache))
    code, document = _json(["--offline"])
    assert code == 0 and "created on first use" in _by_name(document, "Folder")["detail"]
    assert not (tmp_path / "later").exists()
    assert list(tmp_path.iterdir()) == []


def test_the_cache_contents_are_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mapcv.tile_cache import TileCache

    monkeypatch.setenv("MAPCV_CACHE_DIR", str(tmp_path))
    cache = TileCache("https://tiles.example.com/{z}/{x}/{y}.png")
    cache.put(1, 2, 3, b"x" * 1000, (None, None, None, None))
    check = next(c for c in doctor.cache_checks() if c.name == "Contents")
    assert check.detail.startswith("1 tile(s)")


# --- the terminal ------------------------------------------------------------------------


def _terminal(**changes: Any) -> TerminalInfo:
    base: dict[str, Any] = {
        "encoding": "utf-8",
        "is_tty": True,
        "width": 100,
        "color_system": "truecolor",
        "no_color_env": False,
        "no_color_flag": False,
    }
    return TerminalInfo(**{**base, **changes})


def test_terminal_checks_describe_colour_and_encoding() -> None:
    by_name = {c.name: c for c in doctor.terminal_checks(_terminal())}
    assert by_name["Colour"].detail == "on (truecolor)" and by_name["Width"].detail == "100 columns"
    assert by_name["Output"].detail == "an interactive terminal"
    assert by_name["Encoding"].status == "ok"
    expected: list[tuple[dict[str, Any], str]] = [
        ({"no_color_env": True}, "NO_COLOR"),
        ({"no_color_flag": True}, "--no-color"),
        ({"color_system": None, "is_tty": False}, "not a colour terminal"),
    ]
    for changes, text in expected:
        colour = next(c for c in doctor.terminal_checks(_terminal(**changes)) if c.name == "Colour")
        assert colour.detail.startswith("off") and text in colour.detail
    legacy = next(
        c for c in doctor.terminal_checks(_terminal(encoding="cp1252")) if c.name == "Encoding"
    )
    assert legacy.status == "warn" and legacy.hint is not None and "UTF-8" in legacy.hint


def test_no_color_is_reported() -> None:
    result = runner.invoke(app, ["--no-color", "doctor", "--offline"])
    assert "off (--no-color)" in result.output and "\x1b" not in result.output


# --- the network -------------------------------------------------------------------------


def _use_endpoints(monkeypatch: pytest.MonkeyPatch, *endpoints: Endpoint) -> None:
    monkeypatch.setattr(doctor, "default_endpoints", lambda: list(endpoints))


def test_the_default_endpoints_come_from_the_code_that_uses_them() -> None:
    from mapcv.downloader import URL_TEMPLATES

    endpoints = {e.name: e.url for e in doctor.default_endpoints()}
    assert endpoints["Esri World Imagery tiles"] == URL_TEMPLATES["esri_satellite"].format(
        z=0, x=0, y=0
    )
    assert endpoints["EOPF STAC catalog"].startswith("https://stac.core.eopf")
    assert endpoints["Earth Search STAC"].startswith("https://earth-search")
    assert ("Earth Engine" in endpoints) == (importlib.util.find_spec("ee") is not None)


def test_reachable_and_refused_endpoints(monkeypatch: pytest.MonkeyPatch, httpserver: Any) -> None:
    httpserver.expect_request("/up").respond_with_data("{}", content_type="application/json")
    httpserver.expect_request("/root-404").respond_with_data("nope", status=404)
    httpserver.expect_request("/broken").respond_with_data("oops", status=503)
    _use_endpoints(
        monkeypatch,
        Endpoint("Up", httpserver.url_for("/up")),
        Endpoint("Refused", f"http://127.0.0.1:{_refused_port()}/"),
        Endpoint("Not found is fine", httpserver.url_for("/root-404"), accept_client_error=True),
        Endpoint("Not found is not", httpserver.url_for("/root-404")),
        Endpoint("Server error", httpserver.url_for("/broken")),
    )
    started = time.perf_counter()
    code, document = _json([])
    elapsed = time.perf_counter() - started
    assert code == 0 and document["ok"] is True  # an unreachable provider is not a failure
    up = _by_name(document, "Up")
    assert up["status"] == "ok" and "HTTP 200 in" in up["detail"] and " ms" in up["detail"]
    refused = _by_name(document, "Refused")
    assert refused["status"] == "warn" and "connection refused" in refused["detail"]
    assert "HTTPS_PROXY" in refused["hint"]
    assert _by_name(document, "Not found is fine")["status"] == "ok"
    assert _by_name(document, "Not found is not")["status"] == "warn"
    assert _by_name(document, "Server error")["status"] == "warn"
    assert elapsed < 5  # the probes run at once, and nothing waits for a timeout


def test_a_silent_server_is_a_warning_after_the_timeout() -> None:
    with _silent_server() as port:
        started = time.perf_counter()
        checks = doctor.network_checks([Endpoint("Hung", f"http://127.0.0.1:{port}/")], timeout=0.5)
        elapsed = time.perf_counter() - started
    hung = next(c for c in checks if c.name == "Hung")
    assert hung.status == "warn" and "no answer within 0.5 s" in hung.detail
    assert elapsed < 3


def test_the_probes_run_concurrently() -> None:
    with _silent_server() as port:
        endpoints = [Endpoint(f"Hung {i}", f"http://127.0.0.1:{port}/") for i in range(4)]
        started = time.perf_counter()
        doctor.network_checks(endpoints, timeout=0.6)
        elapsed = time.perf_counter() - started
    assert elapsed < 1.8  # four in a row would take 2.4 s


def test_offline_makes_no_request(monkeypatch: pytest.MonkeyPatch, httpserver: Any) -> None:
    _use_endpoints(monkeypatch, Endpoint("Up", httpserver.url_for("/up")))
    code, document = _json(["--offline"])
    assert code == 0 and not [c for c in document["checks"] if c["name"] == "Up"]
    assert httpserver.log == []


def test_earth_engine_is_probed_only_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_modules(monkeypatch, "ee")
    assert "Earth Engine" not in {e.name for e in doctor.default_endpoints()}


# --- no secrets in the output ------------------------------------------------------------


@pytest.fixture
def secret_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home folder whose credential files hold a sentinel, with Earth Engine 'installed'."""
    home = tmp_path / "home"
    (home / ".config" / "earthengine").mkdir(parents=True)
    (home / ".config" / "gcloud").mkdir(parents=True)
    (home / ".config" / "earthengine" / "credentials").write_text(
        json.dumps({"refresh_token": SENTINEL}), encoding="utf-8"
    )
    (home / ".config" / "gcloud" / "application_default_credentials.json").write_text(
        json.dumps({"client_secret": SENTINEL}), encoding="utf-8"
    )
    service_account = tmp_path / "service-account.json"
    service_account.write_text(json.dumps({"private_key": SENTINEL}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(service_account))
    monkeypatch.setenv("HTTPS_PROXY", f"http://user:{SENTINEL}@proxy.invalid:8080")
    real = doctor._installed
    monkeypatch.setattr(doctor, "_installed", lambda module: module == "ee" or real(module))
    return home


def test_credentials_are_reported_by_place_never_by_content(secret_home: Path) -> None:
    code, document = _json(["--offline"])
    credentials = _by_name(document, "Earth Engine credentials")
    assert code == 0 and credentials["status"] == "ok"
    assert "GOOGLE_APPLICATION_CREDENTIALS" in credentials["detail"]
    assert "~/.config/earthengine/credentials".replace("/", os.sep) in credentials["detail"]
    assert "application_default_credentials.json" in credentials["detail"]
    for arguments in (["--json", "--offline"], ["--offline"], ["--json"]):
        result = runner.invoke(app, ["doctor", *arguments])
        assert SENTINEL not in result.output, arguments
        assert "proxy.invalid" not in result.output and str(secret_home.parent) not in result.output


def test_a_proxy_is_named_but_its_value_is_not_shown(secret_home: Path) -> None:
    check = next(c for c in doctor.network_checks([]) if c.name == "Proxy")
    assert "HTTPS_PROXY" in check.detail and SENTINEL not in check.detail


def test_missing_earth_engine_credentials_are_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    real = doctor._installed
    monkeypatch.setattr(doctor, "_installed", lambda module: module == "ee" or real(module))
    check = next(c for c in doctor.extras_checks() if c.name == "Earth Engine credentials")
    assert (
        check.status == "info"
        and check.hint is not None
        and "earthengine authenticate" in check.hint
    )


# --- robustness --------------------------------------------------------------------------


def test_a_crashing_check_becomes_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list[Check]:
        raise RuntimeError("surprise")

    monkeypatch.setattr(doctor, "extras_checks", boom)
    code, document = _json(["--offline"])
    crashed = [c for c in document["checks"] if c["section"] == "Extras"]
    assert code == 0 and crashed[0]["status"] == "warn" and "surprise" in crashed[0]["detail"]


def test_paths_show_the_home_folder_as_a_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert doctor._display_path(tmp_path / "a" / "b") == "~" + os.sep + os.path.join("a", "b")
    assert doctor._display_path(tmp_path.parent / "elsewhere") == str(tmp_path.parent / "elsewhere")
