"""A path that names a pipe or a device is refused with a message, never read for ever."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import ValidationError

from mapcv.agent_tools import Sandbox, ToolState, validate_config
from mapcv.agent_tools import plan as plan_tool
from mapcv.config import MapcvConfig
from mapcv.geotiff import GeoTiff
from mapcv.labels import load_vector_labels
from mapcv.planning import plan

T = TypeVar("T")


needs_fifo = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes here")
needs_zero = pytest.mark.skipif(not Path("/dev/zero").exists(), reason="no /dev/zero here")


def _bounded(call: Callable[[], T], seconds: float = 20.0) -> T:
    """The result of ``call()``, or a failure if it has not returned after ``seconds`` (a
    read of a pipe nobody writes to blocks for ever; it runs in a thread that is left)."""
    outcome: list[Any] = []

    def run() -> None:
        try:
            outcome.append(("ok", call()))
        except BaseException as exc:  # noqa: BLE001 - handed to the test
            outcome.append(("raised", exc))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        pytest.fail(f"still reading after {seconds:g} s: the input was not refused")
    kind, value = outcome[0]
    if kind == "raised":
        raise value
    return value  # type: ignore[no-any-return]


@pytest.fixture(params=["fifo", "zero"])
def special(request: pytest.FixtureRequest, tmp_path: Path) -> Callable[[str], Path]:
    """Makes a named pipe, or a link to /dev/zero, at ``tmp_path / name``."""

    def make(name: str) -> Path:
        path = tmp_path / name
        if request.param == "fifo":
            if not hasattr(os, "mkfifo"):
                pytest.skip("no named pipes here")
            os.mkfifo(path)
        else:
            if not Path("/dev/zero").exists():
                pytest.skip("no /dev/zero here")
            path.symlink_to("/dev/zero")
        return path

    return make


XYZ = {"type": "xyz", "zoom": 16, "source": "esri_satellite"}
REGION = {"west": 4.0, "south": 52.0, "east": 4.02, "north": 52.02}


def _config(**blocks: Any) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": REGION,
            "imagery": XYZ,
            "sampler": {"patch_size": 256},
            "writer": {"staging_dir": "out"},
            **blocks,
        }
    )


def test_a_label_file_that_is_a_pipe_or_device_is_refused(
    special: Callable[[str], Path],
) -> None:
    path = special("labels.geojson")
    with pytest.raises(ValueError, match=r"not a regular file"):
        _bounded(lambda: load_vector_labels(path))
    with pytest.raises(ValueError, match=r"not a regular file"):
        _bounded(lambda: plan(_config(labels={"path": str(path)})))
    kml = path.with_suffix(".kml")
    kml.symlink_to(path)
    with pytest.raises(ValueError, match=r"not a regular file"):
        _bounded(lambda: load_vector_labels(kml))


def test_an_annotated_area_that_is_a_pipe_or_device_is_refused(
    special: Callable[[str], Path],
) -> None:
    from mapcv.labels import label_file_sha256

    with pytest.raises(ValueError, match=r"not a regular file"):
        _bounded(lambda: label_file_sha256(special("area.geojson")))


def test_a_region_file_that_is_a_pipe_or_device_is_refused(
    special: Callable[[str], Path],
) -> None:
    path = special("aoi.geojson")
    with pytest.raises(ValidationError, match=r"region.path .* not a regular file"):
        _bounded(lambda: _config(region={"path": str(path)}))


def test_an_imagery_file_that_is_a_pipe_or_device_is_refused(
    special: Callable[[str], Path],
) -> None:
    path = special("image.tif")
    with pytest.raises(ValueError, match=r"not a regular file"):
        _bounded(lambda: GeoTiff(path))
    config = _config(imagery={"type": "geotiff", "path": str(path)})
    with pytest.raises(ValueError, match=r"not a regular file"):
        _bounded(lambda: plan(config))


def test_a_config_file_that_is_a_pipe_or_device_is_refused(
    special: Callable[[str], Path],
) -> None:
    path = special("mapcv.yaml")
    with pytest.raises(ValueError, match=r"not a regular file"):
        _bounded(lambda: MapcvConfig.from_yaml(path))


def test_the_cli_refuses_a_pipe_or_device_as_config_or_input(
    special: Callable[[str], Path], tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from mapcv.cli import app

    runner = CliRunner()
    config = special("mapcv.yaml")
    result = _bounded(lambda: runner.invoke(app, ["validate", str(config)]))
    assert result.exit_code == 1 and "not a regular file" in " ".join(result.output.split())

    pipe = special("aoi.geojson")
    text = (
        f"region: {{path: {pipe}}}\n"
        "imagery: {type: xyz, zoom: 16, source: esri_satellite}\n"
        "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n"
    )
    (tmp_path / "region.yaml").write_text(text)
    result = _bounded(lambda: runner.invoke(app, ["plan", str(tmp_path / "region.yaml")]))
    assert result.exit_code == 1 and "not a regular file" in " ".join(result.output.split())

    labels = (
        "region: {west: 4.0, south: 52.0, east: 4.02, north: 52.02}\n"
        "imagery: {type: xyz, zoom: 16, source: esri_satellite}\n"
        f"labels: {{path: {pipe}}}\n"
        "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n"
    )
    (tmp_path / "labels.yaml").write_text(labels)
    result = _bounded(lambda: runner.invoke(app, ["plan", str(tmp_path / "labels.yaml")]))
    assert result.exit_code == 1 and "not a regular file" in " ".join(result.output.split())


@needs_fifo
def test_the_mcp_tools_refuse_a_pipe(tmp_path: Path) -> None:
    # (A link to a device is outside the server's folder, which the sandbox refuses already.)
    from mapcv.agent_tools import ToolFailure

    state = ToolState(Sandbox(tmp_path, False))
    pipe = tmp_path / "aoi.geojson"
    os.mkfifo(pipe)
    tail = "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n"
    imagery = "imagery: {type: xyz, zoom: 16, source: esri_satellite}\n"
    region = f"region: {{path: {pipe.name}}}\n" + imagery + tail
    checked = _bounded(lambda: validate_config(state, None, region))
    assert checked.data["valid"] is False
    assert "not a regular file" in str(checked.data["errors"])

    labels = (
        "region: {west: 4.0, south: 52.0, east: 4.02, north: 52.02}\n"
        + imagery
        + f"labels: {{path: {pipe.name}}}\n"
        + tail
    )
    with pytest.raises(ToolFailure, match="not a regular file"):
        _bounded(lambda: plan_tool(state, None, labels))


def test_the_validate_command_refuses_a_pipe_or_device_as_an_input(
    special: Callable[[str], Path], tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from mapcv.cli import app

    runner = CliRunner()
    path = special("labels.geojson")
    text = (
        "region: {west: 4.0, south: 52.0, east: 4.02, north: 52.02}\n"
        "imagery: {type: xyz, zoom: 16, source: esri_satellite}\n"
        f"labels: {{path: {path}}}\n"
        "sampler: {patch_size: 256}\nwriter: {staging_dir: out}\n"
    )
    (tmp_path / "mapcv.yaml").write_text(text)
    result = _bounded(lambda: runner.invoke(app, ["validate", str(tmp_path / "mapcv.yaml")]))
    output = " ".join(result.output.split())
    assert result.exit_code == 1, output
    assert "labels.path" in output and "not a regular file" in output
    assert "is a valid config" not in output

    # An ordinary file, and one that is missing, are as before.
    ordinary = tmp_path / "ordinary.geojson"
    ordinary.write_text('{"type": "FeatureCollection", "features": []}')
    (tmp_path / "ok.yaml").write_text(text.replace(str(path), str(ordinary)))
    result = runner.invoke(app, ["validate", str(tmp_path / "ok.yaml")])
    assert result.exit_code == 0 and "is a valid config" in result.output
    (tmp_path / "gone.yaml").write_text(text.replace(str(path), str(tmp_path / "gone.geojson")))
    result = runner.invoke(app, ["validate", str(tmp_path / "gone.yaml")])
    assert result.exit_code == 0 and "labels.path not found" in " ".join(result.output.split())
