"""``--debug`` and ``--debug-log``: tracebacks after error messages and mapcv's debug log,
for bug reports, without secrets from tile URLs."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from mapcv.cli import app

runner = CliRunner()
REGION = "region: {west: 4.8900, south: 52.3700, east: 4.8990, north: 52.3750}\n"


def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (256, 256), (90, 120, 150)).save(buffer, "PNG")
    return buffer.getvalue()


def _config(tmp_path: Path, template: str, policy: str = "lenient") -> Path:
    path = tmp_path / "mapcv.yaml"
    path.write_text(
        REGION
        + f"imagery: {{type: xyz, zoom: 16, url_template: '{template}', policy: {policy}}}\n"
        + "sampler: {patch_size: 256, edge_strategy: drop}\n"
        + f"writer: {{staging_dir: {tmp_path / 'dataset'}}}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _reset_logging() -> Any:
    """Each test starts and ends without debug handlers (the CLI runs in-process)."""
    yield
    runner.invoke(app, ["--version"])
    from mapcv import cli

    cli._configure_debug(False, None)


def test_a_failure_shows_the_traceback_only_with_debug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError("requested region does not intersect the EOPF product")

    monkeypatch.setattr("mapcv.cli.run_generate", fail)
    config = _config(tmp_path, "https://tiles.example.com/{z}/{x}/{y}.png")
    plain = runner.invoke(app, ["generate", str(config), "--yes"])
    assert plain.exit_code == 1 and "does not intersect" in plain.output
    assert "Traceback" not in plain.output

    debug = runner.invoke(app, ["--debug", "generate", str(config), "--yes"])
    assert debug.exit_code == 1 and "does not intersect" in debug.output
    assert "Traceback (most recent call last)" in debug.output
    assert "ValueError: requested region does not intersect" in debug.output
    assert "mapcv " in debug.output and "Python " in debug.output  # the version header


def test_debug_logs_every_chunk_and_leaves_no_handlers_behind(
    tmp_path: Path, httpserver: Any
) -> None:
    httpserver.expect_request("").respond_with_data(_png(), content_type="image/png")
    template = httpserver.url_for("/tiles/{z}/{x}/{y}.png")
    log_file = tmp_path / "run.log"
    result = runner.invoke(
        app,
        ["--debug-log", str(log_file), "generate", str(_config(tmp_path, template)), "--yes"],
    )
    assert result.exit_code == 0, result.output
    log = log_file.read_text(encoding="utf-8")
    assert "chunk(s) of" in log and "chunk 0:" in log and "fetched" in log
    # --debug-log alone writes the file but keeps the terminal as it was.
    assert "chunk 0:" not in result.output

    runner.invoke(app, ["validate", str(_config(tmp_path, template))])
    logger = logging.getLogger("mapcv")
    assert not [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert logger.level == logging.NOTSET


def test_secrets_in_the_tile_url_stay_out_of_the_debug_log(tmp_path: Path, httpserver: Any) -> None:
    httpserver.expect_request("").respond_with_data(b"denied", status=403)
    template = httpserver.url_for("/v1/SECRETKEY123/{z}/{x}/{y}.png") + "?token=QUERYSECRET"
    log_file = tmp_path / "run.log"
    config = _config(tmp_path, template, policy="strict")
    result = runner.invoke(
        app, ["--debug", "--debug-log", str(log_file), "generate", str(config), "--yes"]
    )
    assert result.exit_code == 1
    text = result.output + log_file.read_text(encoding="utf-8")
    assert "HTTP 403" in text and "Traceback" in text
    assert "SECRETKEY123" not in text and "QUERYSECRET" not in text


@pytest.mark.parametrize(
    ("imagery", "secret"),
    [
        # A typo ({yy}) makes the config invalid; the key sits at the end of the URL.
        (
            (
                "{type: xyz, zoom: 16, url_template: "
                "'http://127.0.0.1:1/{z}/{x}/{yy}.png?api_key=SUPERSECRETKEY42'}"
            ),
            "SUPERSECRETKEY42",
        ),
        # Refused for its credentials, which the error must not then repeat.
        (
            "{type: geotiff, path: 'https://user:PASSWORDSECRET@example.com/x.tif'}",
            "PASSWORDSECRET",
        ),
    ],
)
def test_secrets_of_an_invalid_config_stay_out_of_the_debug_log(
    tmp_path: Path, imagery: str, secret: str
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        REGION + f"imagery: {imagery}\nsampler: {{patch_size: 256}}\n"
        f"writer: {{staging_dir: {tmp_path / 'dataset'}}}\n",
        encoding="utf-8",
    )
    log_file = tmp_path / "debug.log"
    result = runner.invoke(app, ["--debug", "--debug-log", str(log_file), "validate", str(config)])
    assert result.exit_code == 1
    log = log_file.read_text(encoding="utf-8")
    assert "Traceback" in log and "ValidationError" in log
    assert secret not in result.output + log
