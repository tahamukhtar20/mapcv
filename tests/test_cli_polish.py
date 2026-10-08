"""Wizard validation, config rough edges and terminal output found by the CLI tester."""

from __future__ import annotations

import io
import re
import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress
from typer.testing import CliRunner

from mapcv.cli import app
from mapcv.config import EOPFZarrImageryConfig, MapcvConfig
from mapcv.manifest import Manifest, ManifestEntry, PatchSummary, TargetRecord
from mapcv.splitter import SplitterConfig, split_manifest

runner = CliRunner()

_BASE = """\
region: {west: 4.0, south: 52.0, east: 4.02, north: 52.02}
imagery: {type: xyz, zoom: 16, source: esri_satellite}
sampler:
  patch_size: 256
  stride: -5
  mode: grid
  stride: 0
writer: {staging_dir: out}
"""

_POLYGON = (
    '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"kind":"roof"},'
    '"geometry":{"type":"Polygon","coordinates":[[[74.3,31.5],[74.31,31.5],'
    "[74.31,31.51],[74.3,31.5]]]}}]}"
)


def _wizard(tmp_path: Path, answers: list[str]) -> tuple[str, Path]:
    out = tmp_path / "mapcv.yaml"
    result = runner.invoke(
        app, ["init", str(out), "--interactive"], input="\n".join(answers) + "\n"
    )
    return result.output, out


# --- F18, E-9: duplicate keys ---------------------------------------------------------


def test_a_repeated_key_is_an_error_naming_the_key_and_lines(tmp_path: Path) -> None:
    config = tmp_path / "dup.yaml"
    config.write_text(_BASE, encoding="utf-8")
    result = runner.invoke(app, ["validate", str(config)])
    assert result.exit_code == 1
    text = " ".join(result.output.split())
    assert "'stride' twice" in text and "line 7" in text and "line 5" in text
    assert "Traceback" not in result.output
    with pytest.raises(yaml.YAMLError, match="twice"):
        MapcvConfig.from_yaml(config)


def test_repeated_keys_are_found_at_any_depth_and_merge_keys_still_work() -> None:
    from mapcv.config import load_yaml

    with pytest.raises(yaml.YAMLError, match="'a' twice"):
        load_yaml("x:\n  y:\n    a: 1\n    b: 2\n    a: 3\n")
    # the same key in different mappings, and a key overriding a merged one, are fine
    assert load_yaml("x: {a: 1}\ny: {a: 2}\n") == {"x": {"a": 1}, "y": {"a": 2}}
    merged = load_yaml("base: &b {a: 1, b: 2}\nuse:\n  <<: *b\n  a: 9\n")
    assert merged["use"] == {"a": 9, "b": 2}
    assert load_yaml("") is None and load_yaml("- 1\n- 1\n") == [1, 1]


def test_the_mcp_config_path_rejects_repeated_keys(tmp_path: Path) -> None:
    from mapcv.agent_tools import ConfigInvalid, Sandbox, ToolState, parse_config_text

    state = ToolState(Sandbox(tmp_path, False))
    with pytest.raises(ConfigInvalid, match="twice"):
        parse_config_text(state, _BASE, tmp_path)


# --- F8: an empty Sentinel-2 product path ---------------------------------------------


@pytest.mark.parametrize("path", ["", "   "])
def test_an_empty_sentinel2_path_is_invalid(path: str) -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        EOPFZarrImageryConfig(path=path)


# --- F6, F7, F8: the wizard asks again ------------------------------------------------


def test_the_wizard_asks_for_the_zoom_again_when_it_is_out_of_range(tmp_path: Path) -> None:
    answers = ["esri", "74.30,31.48,74.34,31.52", "99", "0", "17", "", "256", "./ds", "n"]
    output, out = _wizard(tmp_path, answers)
    assert output.count("zoom level from 1 to 22") == 2
    assert MapcvConfig.from_yaml(out).imagery.zoom == 17  # type: ignore[union-attr]


def test_the_wizard_asks_again_for_a_missing_or_wrong_label_file(tmp_path: Path) -> None:
    labels = tmp_path / "aoi.geojson"
    labels.write_text(_POLYGON, encoding="utf-8")
    answers = [
        "esri",
        "74.30,31.48,74.34,31.52",
        "17",
        "0",  # not a label file
        str(tmp_path / "missing.geojson"),  # does not exist
        str(labels),
        "kind",
        "",  # segmentation
        "256",
        "./ds",
        "n",
    ]
    output, out = _wizard(tmp_path, answers)
    assert "is not a label file" in output and "File not found" in output
    assert "needs edits" not in output
    config = MapcvConfig.from_yaml(out)
    assert config.labels is not None and Path(str(config.labels.path)) == labels


def test_the_wizard_requires_a_sentinel2_product(tmp_path: Path) -> None:
    answers = [
        "sentinel2",
        "5.40,51.975,5.41,51.98",
        "",
        "ftp://x/y.zarr",
        "S2.zarr",
        "",
        "",
        "",
        "128",
    ]
    output, out = _wizard(tmp_path, answers + ["./ds", "n"])
    assert "A Sentinel-2 product is needed" in output
    assert "imagery.path must be a local path" in output
    assert "mapcv[zarr]" in output
    assert "path: 'S2.zarr'" in out.read_text(encoding="utf-8")


def test_the_wizard_checks_a_tile_url(tmp_path: Path) -> None:
    answers = ["custom", "74.30,31.48,74.34,31.52", "17", "not-a-url"]
    answers += ["https://t.example.com/{z}/{x}/{y}.png", "", "256", "./ds", "n"]
    output, out = _wizard(tmp_path, answers)
    assert "url_template must be an http:// or https:// URL" in output
    assert "needs edits" not in output
    assert MapcvConfig.from_yaml(out).imagery.url_template.endswith("{y}.png")  # type: ignore[union-attr]


def test_the_wizard_checks_earth_engine_dates(tmp_path: Path) -> None:
    answers = ["gee", "74.30,31.48,74.34,31.52", "sentinel2", "tomorrow", "2025-06-01"]
    answers += ["2025-09-01", "", "", "", "17", "", "256", "./ds", "n"]
    output, _ = _wizard(tmp_path, answers)
    assert "Enter a date like 2025-06-01" in output


def test_the_wizard_shows_the_centre_and_the_axis_order_of_a_box(tmp_path: Path) -> None:
    answers = ["esri", "52.37,4.93,52.38,4.95", "17", "", "256", "./ds", "n"]
    output, _ = _wizard(tmp_path, answers)
    assert "Centre: longitude 52.375, latitude 4.94" in output
    assert "longitude first" in output


def test_a_box_that_is_valid_only_as_lat_lon_is_explained(tmp_path: Path) -> None:
    # Tokyo as latitude,longitude: 139.7 is not a latitude, so the first answer is refused
    answers = ["esri", "35.68,139.70,35.70,139.75", "139.70,35.68,139.75,35.70", "17"]
    output, _ = _wizard(tmp_path, answers + ["", "256", "./ds", "n"])
    text = " ".join(output.split())
    assert "look like latitude,longitude" in text
    assert "139.7,35.68,139.75,35.7" in text


def test_the_wizard_says_where_change_and_regression_start(tmp_path: Path) -> None:
    output, _ = _wizard(tmp_path, ["esri"])  # the input ends after the first answer
    assert "--template change" in output and "--template regression" in output


# --- F16: no terminal -----------------------------------------------------------------


def test_init_without_a_terminal_says_it_wrote_a_template(tmp_path: Path) -> None:
    piped = tmp_path / "piped.yaml"
    result = runner.invoke(app, ["init", str(piped)])  # CliRunner stdin is not a terminal
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "No terminal was found" in text and "xyz template" in text and "--template" in text
    flag = tmp_path / "flag.yaml"
    result = runner.invoke(app, ["init", str(flag), "--no-interactive"])
    assert "--no-interactive is set" in " ".join(result.output.split())
    named = tmp_path / "named.yaml"
    result = runner.invoke(app, ["init", str(named), "--template", "xyz"])
    assert "no questions were asked" not in " ".join(result.output.split())


# --- F12: progress on a narrow terminal -----------------------------------------------


@pytest.mark.parametrize("width", [40, 50, 64, 70, 80, 100, 120])
def test_the_progress_line_fits_without_cutting_a_column(width: int) -> None:
    from mapcv.cli import _progress_columns

    stream = io.StringIO()
    console = Console(file=stream, width=width, force_terminal=True, color_system=None)
    with Progress(*_progress_columns(width), console=console) as progress:
        task = progress.add_task("Reading imagery and writing patches", total=4)
        progress.update(task, completed=1)
    frame = [part for part in stream.getvalue().split("\r") if "1/4" in part][-1]
    line = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", frame).split("\n")[0]
    assert "…" not in line and "1/4" in line
    assert len(line) <= width


def test_the_progress_line_keeps_elapsed_and_eta_at_70_columns() -> None:
    from mapcv.cli import _progress_columns

    stream = io.StringIO()
    console = Console(file=stream, width=70, force_terminal=True, color_system=None)
    with Progress(*_progress_columns(70), console=console) as progress:
        progress.add_task("x", total=4)
    assert "eta" in stream.getvalue() and "0:00:00" in stream.getvalue()


# --- F17: an empty split --------------------------------------------------------------


def _region_manifest(regions: int, per_region: int = 3) -> Manifest:
    manifest = Manifest(
        target=TargetRecord(type="segmentation", class_map={"bg": 0, "obj": 1}),
        sampler={"patch_size": 1, "stride": 1},
    )
    index = 0
    for region in range(regions):
        for _ in range(per_region):
            summary = PatchSummary(empty_ratio=0.0)
            summary["region"] = f"r{region}"
            manifest.patches.append(
                ManifestEntry(
                    row=index * 10,  # far apart: nothing overlaps
                    col=0,
                    padded=False,
                    chunk=0,
                    files={"image": f"Images/patch_{index:07d}.png"},
                    summary=summary,
                )
            )
            index += 1
    return manifest


def test_a_region_split_with_an_empty_val_split_warns(tmp_path: Path) -> None:
    config = SplitterConfig(strategy="region")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        counts, _ = split_manifest(_region_manifest(3), config, tmp_path / "splits")
    assert counts["val"] == 0 and counts["train"] > 0
    messages = [str(item.message) for item in caught]
    assert any(
        "No patch is left for val" in message and "regions" in message for message in messages
    )


def test_no_warning_when_every_requested_split_has_patches(tmp_path: Path) -> None:
    config = SplitterConfig(strategy="random", test_ratio=0.2, val_ratio=0.1)
    manifest = _region_manifest(1, per_region=40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        counts, _ = split_manifest(manifest, config, tmp_path / "splits")
    assert counts["val"] > 0 and counts["test"] > 0
    assert not [item for item in caught if "No patch is left" in str(item.message)]
    config = SplitterConfig(strategy="random", test_ratio=0.2, val_ratio=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        split_manifest(manifest, config, tmp_path / "splits2")
    assert not [item for item in caught if "val" in str(item.message)]


# --- F3: --sample-limit ---------------------------------------------------------------


def test_split_sample_limit_says_how_many_patches_are_in_no_split(tmp_path: Path) -> None:
    staging = tmp_path / "ds"
    staging.mkdir()
    _region_manifest(1, per_region=30).save(staging / "manifest.json")
    result = runner.invoke(
        app, ["split", str(staging), "--strategy", "random", "--sample-limit", "10"]
    )
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "20 patches of 30 are in no split" in text and "--sample-limit 10" in text
    result = runner.invoke(app, ["split", str(staging), "--strategy", "random"])
    assert "in no split" not in result.output


def test_split_help_explains_sample_limit() -> None:
    result = runner.invoke(app, ["split", "--help"])
    # the help sits in a box whose border characters differ between platforms
    assert "nosplit" in re.sub(r"[^a-z]+", "", result.output)


# --- F4: extra names ------------------------------------------------------------------


def test_doctor_names_both_pyarrow_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, package=None: None if name == "pyarrow" else real(name, package),
    )
    from mapcv.doctor import extras_checks

    check = next(item for item in extras_checks() if item.name == "parquet / export")
    assert "mapcv[parquet]" in (check.hint or "") and "mapcv[export]" in (check.hint or "")
