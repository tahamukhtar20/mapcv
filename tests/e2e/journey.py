"""End-to-end user journey against an installed mapcv, run by the release workflow.

Runs the real ``mapcv`` command (not the test runner's in-process CLI) through
init (answering the wizard on stdin), validate, plan, generate, resume, info and
split, in a folder whose name has a space and a non-ASCII character, against a
local tile server. Exits non-zero with the failing step's output on any problem.

    python tests/e2e/journey.py --version 0.2.0
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional

from PIL import Image

ZOOM = 18
X0, Y0, NX, NY = 134_700, 86_100, 6, 5  # 30 tiles near Amsterdam
# No PYTHONIOENCODING: on Windows redirected output uses the locale code page (cp1252),
# which is exactly what a user piping `mapcv plan > plan.txt` gets.
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
ENV.update(COLUMNS="120", NO_COLOR="1")


def lon(x: float) -> float:
    return float(x / 2**ZOOM * 360 - 180)


def lat(y: float) -> float:
    return float(math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2**ZOOM)))))


class Tiles(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        z, x, y = (int(part) for part in self.path.strip("/").split(".")[0].split("/"))
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256), ((x * 37) % 256, (y * 53) % 256, z * 9)).save(buffer, "PNG")
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(args: List[str], cwd: Path, stdin: Optional[str] = None, expect: int = 0) -> str:
    print(f"$ mapcv {' '.join(args)}", flush=True)
    result = subprocess.run(
        ["mapcv", *args],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=ENV,
        timeout=600,
    )
    output = result.stdout + result.stderr
    if result.returncode != expect:
        sys.exit(
            f"`mapcv {' '.join(args)}` exited {result.returncode}, expected {expect}:\n{output}"
        )
    return output


def check(condition: bool, message: str, output: str = "") -> None:
    if not condition:
        sys.exit(f"FAILED: {message}\n{output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    version = parser.parse_args().version

    ThreadingHTTPServer.request_queue_size = 128
    server = ThreadingHTTPServer(("127.0.0.1", 0), Tiles)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    template = f"http://127.0.0.1:{server.server_address[1]}/{{z}}/{{x}}/{{y}}.png"

    root = Path(tempfile.mkdtemp()) / "mapcv e2e ü"
    root.mkdir()
    eps = 1e-7
    west, east = lon(X0) + eps, lon(X0 + NX) - eps
    north, south = lat(Y0) - eps, lat(Y0 + NY) + eps
    w, h = (east - west) / 4, (north - south) / 4
    features = [
        {
            "type": "Feature",
            "properties": {"class": name},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]],
            },
        }
        for name, x, y in (
            ("building", west + w / 2, south + h / 2),
            ("water", west + 2.5 * w, south + 2 * h),
        )
    ]
    (root / "labels.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )

    out = run(["--version"], root)
    check(out.strip() == f"mapcv {version}", f"--version printed {out.strip()!r}", out)

    answers = "\n".join(
        [
            "custom",
            f"{west},{south},{east},{north}",
            str(ZOOM),
            template,
            "labels.geojson",
            "class",
            "",  # task: default segmentation
            "",  # patch size: default 256
            "",  # output folder: default ./dataset
            "",  # split: default yes
        ]
    )
    out = run(["init", "--interactive"], root, stdin=answers + "\n")
    config = root / "mapcv.yaml"
    check(config.exists(), "init did not write mapcv.yaml", out)
    text = config.read_text(encoding="utf-8")
    check("url_template" in text and "label_field: class" in text, "unexpected config", text)

    run(["validate", "mapcv.yaml"], root)
    out = run(["plan", "mapcv.yaml"], root)
    check(f"{NX * NY} tiles" in out, "plan did not report the tile count", out)

    # Run from another folder: paths in the config resolve against its own folder.
    out = run(["generate", str(config), "--yes"], root.parent)
    dataset = root / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    patches = manifest["patches"]
    check(len(patches) == NX * NY, f"{len(patches)} patches, expected {NX * NY}", out)
    check(manifest["version"] == 3 and manifest["task"] == "segmentation", "not a v3 manifest", out)
    class_map = manifest["target"]["class_map"]
    check(class_map == {"building": 1, "water": 2}, str(class_map), out)
    images = sorted(f"Images/{p.name}" for p in (dataset / "Images").iterdir())
    masks = sorted(f"Masks/{p.name}" for p in (dataset / "Masks").iterdir())
    check(images == sorted(p["files"]["image"] for p in patches), "Images/ != manifest", out)
    check(masks == sorted(p["files"]["mask"] for p in patches), "Masks/ != manifest", out)
    labeled = sum(1 for p in patches if set(p["summary"]["class_pixels"]) - {"0", "255"})
    check(labeled > 0, "no patch contains a label", out)
    splits = {
        name: (dataset / "splits" / f"{name}.txt").read_text(encoding="utf-8").split()
        for name in ("train", "val", "test")
    }
    check(sum(map(len, splits.values())) > 0, "empty splits", out)

    out = run(["generate", "mapcv.yaml", "--yes"], root)
    check("Nothing left to do" in out, "a finished run did not resume as a no-op", out)

    # Resume after losing the last chunk: the result must equal the full run.
    partial = dict(manifest, patches=patches[: len(patches) // 2])
    (dataset / "manifest.json").write_text(json.dumps(partial), encoding="utf-8")
    out = run(["generate", "mapcv.yaml", "--yes"], root)
    resumed = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))["patches"]
    check(resumed == patches, "resumed manifest differs from the uninterrupted one", out)

    out = run(["info", "dataset"], root)
    check(f"{NX * NY}" in out and "building" in out, "info output incomplete", out)
    out = run(["split", "dataset", "--test-ratio", "0.25"], root)
    check("Splits written" in out, "split did not report", out)

    # The same area as an object detection dataset: COCO and YOLO boxes.
    boxes_config = root / "boxes.yaml"
    boxes_config.write_text(
        "task: detection\n" + text.replace("staging_dir: './dataset'", "staging_dir: './boxes'"),
        encoding="utf-8",
    )
    out = run(["generate", "boxes.yaml", "--yes"], root)
    boxes = root / "boxes"
    manifest = json.loads((boxes / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["task"] == "detection", "not a detection manifest", out)
    patches = manifest["patches"]
    check(len(patches) == NX * NY, f"{len(patches)} detection patches", out)
    objects = sum(sum(p["summary"]["class_objects"].values()) for p in patches)
    check(objects > 0, "no objects in the detection dataset", out)
    images = sorted(f"images/{p.name}" for p in (boxes / "images").iterdir())
    check(images == sorted(p["files"]["image"] for p in patches), "images/ != manifest", out)
    lines = 0
    for name in ("train", "val", "test"):
        coco = json.loads(
            (boxes / "annotations" / f"instances_{name}.json").read_text(encoding="utf-8")
        )
        listed = (boxes / "splits" / f"{name}.txt").read_text(encoding="utf-8").split()
        check(sorted(i["file_name"] for i in coco["images"]) == sorted(listed), name, out)
        lines += len(coco["annotations"])
    labels = sum(
        len(p.read_text(encoding="utf-8").splitlines()) for p in (boxes / "labels").iterdir()
    )
    check(labels >= lines > 0, f"{labels} YOLO lines for {lines} split COCO boxes", out)
    dataset_yaml = (boxes / "dataset.yaml").read_text(encoding="utf-8")
    check("names:" in dataset_yaml and "train: train.txt" in dataset_yaml, dataset_yaml, out)
    out = run(["info", "boxes"], root)
    check("objects" in out and "building" in out, "info shows no object counts", out)
    out = run(["generate", "boxes.yaml", "--yes"], root)
    check("Nothing left to do" in out, "a finished detection run did not resume", out)

    out = run(["generate", "missing.yaml"], root, expect=1)
    check("Traceback" not in out, "a user error printed a traceback", out)

    server.shutdown()
    shutil.rmtree(root.parent, ignore_errors=True)
    print(f"e2e journey passed on {sys.platform} (Python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()
