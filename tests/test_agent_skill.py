"""The agent skill and the Claude Code plugin files stay valid as the config evolves."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from mapcv.config import MapcvConfig

_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _ROOT / "agent" / "skills" / "mapcv" / "SKILL.md"
_PLUGIN = _ROOT / "agent" / ".claude-plugin" / "plugin.json"
_MARKETPLACE = _ROOT / ".claude-plugin" / "marketplace.json"

pytestmark = pytest.mark.skipif(
    not _SKILL.exists(), reason="not a source checkout (the sdist leaves out agent/)"
)

_BASE: Dict[str, Any] = {
    "region": {"west": 4.9375, "south": 52.3725, "east": 4.9515, "north": 52.3780},
    "imagery": {"type": "xyz", "zoom": 18, "source": "esri_satellite"},
    "sampler": {"patch_size": 256},
    "writer": {"staging_dir": "dataset"},
}


def _recipes() -> List[str]:
    text = _SKILL.read_text(encoding="utf-8")
    return re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)


def test_skill_has_front_matter_and_recipes() -> None:
    text = _SKILL.read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    assert front["name"] == "mapcv"
    assert len(front["description"]) > 80
    assert len(_recipes()) >= 6


def test_every_recipe_in_the_skill_validates() -> None:
    """Each recipe, merged over the plain XYZ segmentation config, is a valid config."""
    for recipe in _recipes():
        merged = {**_BASE, **yaml.safe_load(recipe)}
        if merged.get("task") in ("detection", "instance") or "labels" in merged:
            merged.setdefault("labels", {"path": "labels.geojson"})
        try:
            MapcvConfig.model_validate(merged)
        except Exception as exc:  # noqa: BLE001 - name the recipe that broke
            pytest.fail(f"recipe does not validate:\n{recipe}\n{exc}")


def test_plugin_manifests_follow_the_documented_shape() -> None:
    plugin = json.loads(_PLUGIN.read_text(encoding="utf-8"))
    marketplace = json.loads(_MARKETPLACE.read_text(encoding="utf-8"))
    assert plugin["name"] == "mapcv" and plugin["author"]["name"]
    assert not plugin["name"].startswith(("claude", "anthropic"))
    server = plugin["mcpServers"]["mapcv"]
    assert server["command"] == "uvx"
    assert server["args"] == ["--from", "mapcv[mcp]", "mapcv", "mcp"]
    assert "${user_config.allow_write}" in server["env"]["MAPCV_MCP_ALLOW_WRITE"]
    assert set(plugin["userConfig"]["allow_write"]) <= {
        "type",
        "title",
        "description",
        "default",
        "required",
    }
    assert marketplace["name"] and marketplace["owner"]["name"]
    (entry,) = marketplace["plugins"]
    assert entry["name"] == plugin["name"]  # the install id and the manifest name agree
    source = entry["source"]
    assert source.startswith("./") and ".." not in source
    assert (_ROOT / source / ".claude-plugin" / "plugin.json").exists()
    assert (_ROOT / source / "skills" / "mapcv" / "SKILL.md").exists()


def test_the_agent_files_stay_out_of_the_sdist() -> None:
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '{ path = "agent/**/*", format = "sdist" }' in pyproject
    assert '{ path = ".claude-plugin/**/*", format = "sdist" }' in pyproject
