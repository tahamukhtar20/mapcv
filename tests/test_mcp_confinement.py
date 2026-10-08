"""The MCP server keeps to its root and its credentials, however a config, a dataset
folder or a catalog is laid out."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")
pytest.importorskip("rasterio", reason="the GeoTIFF is written with rasterio")

from mcp.client import Client
from test_confinement import _dataset, _item, catalog  # noqa: F401 - catalog is a fixture
from test_mcp_server import call, data, run_client, text_of

REGION = "region: {west: 4.89, south: 52.37, east: 4.899, north: 52.375}\n"
REST = "sampler: {patch_size: 256}\nwriter: {staging_dir: dataset}\n"


def test_a_shapefile_sidecar_outside_the_root_is_not_read(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("OUTSIDE-FILE-CONTENT\n")
    (root / "aoi.shp").write_bytes(b"\0" * 200)
    (root / "aoi.prj").symlink_to(tmp_path / "outside.txt")
    text = "region: {path: aoi.shp}\nimagery: {type: xyz, source: esri, zoom: 17}\n" + REST

    async def scenario(client: Client) -> str:
        return text_of(await call(client, "validate_config", yaml_text=text))

    shown = run_client(root, scenario, write=False)
    assert "OUTSIDE-FILE-CONTENT" not in shown
    assert "region.path (sidecar)" in shown and "outside the folder" in shown


def test_split_does_not_write_through_a_hard_link(tmp_path: Path) -> None:
    root = tmp_path / "root"
    staging = _dataset(root)
    victim = tmp_path / "victim.txt"
    victim.write_text("VICTIM ORIGINAL CONTENT\n")
    train = staging / "splits" / "train.txt"
    train.unlink()
    os.link(victim, train)

    async def scenario(client: Client) -> str:
        result = await call(client, "split", dataset="dataset", test_ratio=0.3)
        assert result.is_error
        return text_of(result)

    shown = run_client(root, scenario)
    assert "hard link" in shown and "splits/train.txt" in shown
    assert victim.read_text() == "VICTIM ORIGINAL CONTENT\n"


def test_config_text_written_as_json_is_a_config(tmp_path: Path) -> None:
    """JSON is YAML: the SDK must not turn it into an object and reject it, quoting the key."""
    config = {
        "region": {"west": 4.89, "south": 52.37, "east": 4.899, "north": 52.375},
        "imagery": {
            "type": "xyz",
            "zoom": 17,
            "url_template": "https://tiles.example.com/{z}/{x}/{y}.png?k=SECRETKEY99",
        },
        "sampler": {"patch_size": 256},
        "writer": {"staging_dir": "dataset"},
    }

    async def scenario(client: Client) -> list[Any]:
        validated = await call(client, "validate_config", yaml_text=json.dumps(config))
        # An argument of the wrong type fails in the SDK, before the tool: redacted too.
        typed = await call(client, "write_config", path=config, yaml_text="x: 1")
        return [validated, typed]

    validated, typed = run_client(tmp_path, scenario)
    assert not validated.is_error and data(validated)["valid"] is True
    assert typed.is_error
    for result in (validated, typed):
        assert "SECRETKEY99" not in text_of(result)


def test_a_url_template_on_a_metadata_address_is_flagged(tmp_path: Path) -> None:
    text = (
        REGION
        + "imagery: {type: xyz, zoom: 17, url_template: 'http://169.254.169.254/{z}/{x}/{y}.png'}\n"
        + REST
    )
    ordinary = REGION + "imagery: {type: xyz, source: esri, zoom: 17}\n" + REST

    async def scenario(client: Client) -> list[dict[str, Any]]:
        return [
            data(await call(client, "validate_config", yaml_text=text)),
            data(await call(client, "validate_config", yaml_text=ordinary)),
        ]

    flagged, plain = run_client(tmp_path, scenario, write=False)
    assert flagged["valid"] is True  # a warning, not a refusal
    assert any("169.254.169.254" in warning for warning in flagged["warnings"])
    assert not any("metadata" in warning for warning in plain["warnings"])


def test_a_catalog_cannot_point_outside_the_root(
    tmp_path: Path,
    catalog: dict[str, Any],  # noqa: F811 - the fixture imported above
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.tif"
    outside.write_bytes(b"II*\0")
    catalog["page"] = {"features": [_item(str(outside))], "links": []}
    text = (
        "region: {west: 74.301, south: 31.441, east: 74.33, north: 31.489}\n"
        f"imagery: {{type: stac_cog, search: {{catalog: '{catalog['url']}', "
        "datetime: '2024-06-01'}, bands: [red, green, blue]}\n"
        "sampler: {patch_size: 64}\nwriter: {staging_dir: dataset, image_format: npy}\n"
    )

    (root / "stac.yaml").write_text(text, encoding="utf-8")

    async def scenario(client: Client) -> str:
        result = await call(client, "generate", config="stac.yaml")
        assert result.is_error
        return text_of(result)

    shown = run_client(root, scenario)
    assert "a file named by the STAC catalog" in shown and "outside the folder" in shown
    assert not (root / "dataset" / "Images").exists()
