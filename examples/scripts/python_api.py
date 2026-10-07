"""Use mapcv from Python: build a config, estimate it, then generate it.

This is the quickstart (examples/quickstart/mapcv.yaml) without a YAML file.
The same dict is what ``yaml.safe_load`` returns for that file, so you can also
load a config with ``MapcvConfig.from_yaml(path)`` and tweak it in code.

    python examples/scripts/python_api.py                 # plan, then generate
    python examples/scripts/python_api.py --plan-only     # no download

Imagery: Esri World Imagery (check Esri's terms before sharing the output).
Labels: © OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mapcv
from mapcv import MapcvConfig

QUICKSTART = Path(__file__).resolve().parent.parent / "quickstart"


def build_config(staging_dir: Path) -> MapcvConfig:
    """The quickstart config as a plain dict, validated into a MapcvConfig."""
    raw: dict[str, Any] = {
        "region": {"west": 4.9375, "south": 52.3725, "east": 4.9515, "north": 52.3780},
        "imagery": {"type": "xyz", "zoom": 18, "source": "esri_satellite", "max_connections": 4},
        "labels": {"path": QUICKSTART / "buildings.geojson", "label_field": "class"},
        "sampler": {"patch_size": 256, "edge_strategy": "drop"},
        "writer": {"staging_dir": staging_dir, "image_format": "png"},
        "split": {"strategy": "spatial", "test_ratio": 0.2, "val_ratio": 0.1, "seed": 42},
    }
    # Raises pydantic.ValidationError with a readable message if anything is off.
    return MapcvConfig.model_validate(raw)


def print_plan(config: MapcvConfig) -> None:
    """Estimate the run without downloading anything."""
    estimate = mapcv.plan(config)
    width, height = estimate.raster_px
    print("Plan")
    print(f"  region      {estimate.region_km[0]:.2f} x {estimate.region_km[1]:.2f} km")
    print(f"  imagery     {estimate.imagery} at {estimate.resolution_m:.2f} m/px")
    print(f"  raster      {width} x {height} px, {estimate.tiles} tiles")
    print(f"  patches     {estimate.patches} x {estimate.patch_size} px")
    print(f"  download    ~{(estimate.download_bytes or 0) / 1e6:.1f} MB")
    print(f"  output      ~{estimate.output_bytes / 1e6:.1f} MB")
    if estimate.labels is not None:
        print(
            f"  labels      {estimate.labels.polygons} polygons, classes {estimate.labels.classes}"
        )
    for warning in estimate.warnings:
        print(f"  warning     {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=Path("dataset-python-api"), help="output directory"
    )
    parser.add_argument("--plan-only", action="store_true", help="estimate only; no download")
    args = parser.parse_args()

    config = build_config(args.out)
    print_plan(config)
    if args.plan_only:
        return

    # Prints nothing itself; resumes if interrupted. progress(done, total) counts chunks.
    result = mapcv.generate(config, progress=lambda done, total: print(f"  chunk {done}/{total}"))
    print("\nGenerateResult")
    print(f"  staging_dir      {result.staging_dir}")
    print(f"  new_patches      {result.new_patches} (manifest: {len(result.manifest.patches)})")
    print(f"  split_counts     {result.split_counts}")
    print(f"  tiles            {result.tiles_requested} requested, {result.tiles_failed} failed")
    print(f"  seconds          {result.seconds:.1f}")
    manifest = result.manifest  # a mapcv.Manifest (manifest.json, version 3)
    print(f"  task             {manifest.task}")
    print(f"  class_map        {manifest.class_map}")
    print(f"  patch_shape      {manifest.source.patch_shape} {manifest.source.dtype}")
    if manifest.patches:
        first = manifest.patches[0]
        print(f"  first patch      {first['files']}, bounds {manifest.patch_bounds(first)}")


if __name__ == "__main__":
    main()
