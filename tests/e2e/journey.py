"""End-to-end user journey against an installed mapcv, run by the release workflow.

Runs the real ``mapcv`` command (not the test runner's in-process CLI) through
init (answering the wizard on stdin), validate, plan, generate, resume, info and
split (segmentation, detection, instance segmentation and classification datasets), in a folder whose name has a space and a non-ASCII character, against a
local tile server. Exits non-zero with the failing step's output on any problem.

    python tests/e2e/journey.py --version 0.2.0
"""

from __future__ import annotations

import argparse
import csv
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

import numpy as np
import yaml
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

    # The same area as an instance segmentation dataset: COCO RLE masks and instance-ID PNGs.
    masks_config = root / "masks.yaml"
    masks_config.write_text(
        "task: instance\n"
        + text.replace("staging_dir: './dataset'", "staging_dir: './masks'")
        + "\ninstance:\n  id_mask: true\n",
        encoding="utf-8",
    )
    out = run(["generate", "masks.yaml", "--yes"], root)
    inst = root / "masks"
    manifest = json.loads((inst / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["task"] == "instance", "not an instance manifest", out)
    patches = manifest["patches"]
    check(len(patches) == NX * NY, f"{len(patches)} instance patches", out)
    check(
        sorted(f"masks/{p.name}" for p in (inst / "masks").iterdir())
        == sorted(p["files"]["mask"] for p in patches),
        "masks/ != manifest",
        out,
    )
    instances = 0
    for name in ("train", "val", "test"):
        coco = json.loads(
            (inst / "annotations" / f"instances_{name}.json").read_text(encoding="utf-8")
        )
        for image in coco["images"]:
            entry = patches[image["id"] - 1]
            ids = np.array(Image.open(inst / entry["files"]["mask"]))
            check(
                ids.dtype == np.uint16 and ids.shape == (256, 256), f"{ids.dtype} {ids.shape}", out
            )
            found = [ann for ann in coco["annotations"] if ann["image_id"] == image["id"]]
            # The labels do not overlap, so annotation k owns exactly the pixels numbered k.
            check(int(ids.max()) == len(found), f"{ids.max()} ids, {len(found)} annotations", out)
            for number, ann in enumerate(found, start=1):
                rle = ann["segmentation"]
                check(rle["size"] == [256, 256] and isinstance(rle["counts"], str), str(rle), out)
                check(ann["area"] == int((ids == number).sum()), f"area of {ann['id']}", out)
                instances += 1
    counted = sum(sum(p["summary"]["class_objects"].values()) for p in patches)
    check(instances == counted > 0, f"{instances} COCO instances, {counted} in the manifest", out)
    out = run(["info", "masks"], root)
    check("objects" in out and "building" in out, "info shows no instance counts", out)
    out = run(["generate", "masks.yaml", "--yes"], root)
    check("Nothing left to do" in out, "a finished instance run did not resume", out)

    # The same area as a classification dataset: a label per patch from the label coverage.
    scenes_config = root / "scenes.yaml"
    scenes_config.write_text(
        "task: classification\n"
        + text.replace("staging_dir: './dataset'", "staging_dir: './scenes'")
        + "\nclassification:\n  mode: multi\n  empty: background\n",
        encoding="utf-8",
    )
    out = run(["generate", "scenes.yaml", "--yes"], root)
    scenes = root / "scenes"
    manifest = json.loads((scenes / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["task"] == "classification", "not a classification manifest", out)
    patches = manifest["patches"]
    check(len(patches) == NX * NY, f"{len(patches)} classification patches", out)
    images = sorted(f"images/{p.name}" for p in (scenes / "images").iterdir())
    check(images == sorted(p["files"]["image"] for p in patches), "images/ != manifest", out)
    classes = (scenes / "classes.txt").read_text(encoding="utf-8").split("\n")
    check(classes == ["background", "building", "water", ""], f"classes.txt: {classes}", out)
    with (scenes / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) == len(patches), f"{len(rows)} rows in labels.csv", out)
    seen = {label for row in rows for label in row["labels"].split()}
    check(seen == {"background", "building", "water"}, f"labels {seen}", out)
    check(
        json.loads((scenes / "labels.json").read_text(encoding="utf-8"))["images"]
        == {row["image"]: row["labels"].split() for row in rows},
        "labels.json != labels.csv",
        out,
    )
    for name in ("train", "val", "test"):
        listed = (scenes / "splits" / f"{name}.txt").read_text(encoding="utf-8").split()
        with (scenes / f"labels_{name}.csv").open(newline="", encoding="utf-8") as handle:
            per_split = [row["image"] for row in csv.DictReader(handle)]
        check(sorted(per_split) == sorted(listed), f"labels_{name}.csv != {name}.txt", out)
        check(
            sorted(row["image"] for row in rows if row["split"] == name) == sorted(listed),
            f"labels.csv split column != {name}.txt",
            out,
        )
    out = run(["info", "scenes"], root)
    check("patches" in out and "building" in out and "background" in out, "info", out)
    out = run(["split", "scenes", "--test-ratio", "0.4"], root)
    with (scenes / "labels_test.csv").open(newline="", encoding="utf-8") as handle:
        retested = sorted(row["image"] for row in csv.DictReader(handle))
    listed = (scenes / "splits" / "test.txt").read_text(encoding="utf-8").split()
    check(
        retested == sorted(listed) and len(listed) > 0, "split did not rewrite labels_test.csv", out
    )
    out = run(["generate", "scenes.yaml", "--yes"], root)
    check("Nothing left to do" in out, "a finished classification run did not resume", out)

    # Two sources on one grid: the same area at zoom 18 and at zoom 17 (2x coarser pixels).
    two_data = yaml.safe_load(text)
    xyz = {key: value for key, value in two_data["imagery"].items() if key != "zoom"}
    two_data["imagery"] = [
        {**xyz, "name": "z18", "zoom": ZOOM},
        {**xyz, "name": "z17", "zoom": ZOOM - 1},
    ]
    two_data["writer"]["staging_dir"] = "./two"
    # Patches that do not line up with the tiles, so they cross zoom-17 tile edges.
    two_data["sampler"]["patch_size"] = 192
    two_data["sampler"].pop("stride", None)
    (root / "two.yaml").write_text(yaml.safe_dump(two_data, sort_keys=False), encoding="utf-8")
    out = run(["generate", "two.yaml", "--yes"], root)
    two = root / "two"
    manifest = json.loads((two / "manifest.json").read_text(encoding="utf-8"))
    check([s["name"] for s in manifest["sources"]] == ["z18", "z17"], "two sources", out)
    check(manifest["sources"][1].get("factor") == 2, "z17 is not 2x coarser", out)
    patches = manifest["patches"]
    check(len(patches) > NX * NY, f"{len(patches)} two-source patches", out)
    for name in ("z18", "z17"):
        names = sorted(f"Images/{name}/{p.name}" for p in (two / "Images" / name).iterdir())
        check(names == sorted(p["files"][name] for p in patches), f"Images/{name} != manifest", out)
    # Every tile is one colour (see Tiles), so each pixel of a z17 patch must show the
    # colour of the zoom-17 tile containing it: z18 pixel p lies in z17 pixel p // 2.
    a, _, c, _, _, f = manifest["sources"][0]["transform"]
    half_world = 20037508.342789244
    col0, row0 = round((c + half_world) / a), round((half_world - f) / a)
    size = manifest["sampler"]["patch_size"]
    edges = 0
    for entry in patches:
        pixels = np.asarray(Image.open(two / entry["files"]["z17"]).convert("RGB"))
        cols = (col0 + entry["col"] + np.arange(size)) // 2 // 256
        rows = (row0 + entry["row"] + np.arange(size)) // 2 // 256
        want = np.zeros((size, size, 3), dtype=np.uint8)
        want[..., 0] = ((cols * 37) % 256)[np.newaxis, :]
        want[..., 1] = ((rows * 53) % 256)[:, np.newaxis]
        want[..., 2] = (ZOOM - 1) * 9
        # Beyond the raster (NY x NX tiles of 256 px) the patch is padded with zeros.
        want[NY * 256 - entry["row"] :, :] = 0
        want[:, NX * 256 - entry["col"] :] = 0
        check(np.array_equal(pixels, want), f"z17 pixels at {entry['row']},{entry['col']}", out)
        edges += int(len(set(cols)) > 1) + int(len(set(rows)) > 1)
    check(edges > 0, "no z17 patch crosses a zoom-17 tile edge: the check proves nothing", out)
    out = run(["info", "two"], root)
    check("Source z17" in out and "coarser" in out, "info does not list both sources", out)

    # Change detection: the same area at zoom 18 twice (before, after), the labels as change.
    pair_data = yaml.safe_load(text)
    xyz = dict(pair_data["imagery"])
    pair_data["task"] = "change"
    pair_data["imagery"] = [{**xyz, "name": "before"}, {**xyz, "name": "after"}]
    pair_data["writer"]["staging_dir"] = "./pairs"
    (root / "pairs.yaml").write_text(yaml.safe_dump(pair_data, sort_keys=False), encoding="utf-8")
    out = run(["generate", "pairs.yaml", "--yes"], root)
    pairs = root / "pairs"
    manifest = json.loads((pairs / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["task"] == "change" and manifest["writer"]["layout"] == "change", "change", out)
    patches = manifest["patches"]
    check(len(patches) == NX * NY, f"{len(patches)} change patches", out)
    for folder, key in (("A", "before"), ("B", "after"), ("label", "mask")):
        names = sorted(f"{folder}/{p.name}" for p in (pairs / folder).iterdir())
        check(names == sorted(p["files"][key] for p in patches), f"{folder}/ != manifest", out)
    values = set()
    for entry in patches:
        values |= set(np.unique(np.asarray(Image.open(pairs / entry["files"]["mask"]))).tolist())
    check(1 in values and values <= {0, 1, 255}, f"change mask values {sorted(values)}", out)
    out = run(["info", "pairs"], root)
    check("change" in out and "Source after" in out, "info does not describe the pairs", out)

    # Two dates stacked into one GeoTIFF per patch (writer.stack_sources).
    stack_data = yaml.safe_load(text)
    xyz = dict(stack_data["imagery"])
    stack_data["imagery"] = [{**xyz, "name": "t0"}, {**xyz, "name": "t1"}]
    stack_data["writer"].update(staging_dir="./stacks", image_format="tif", stack_sources=True)
    (root / "stacks.yaml").write_text(yaml.safe_dump(stack_data, sort_keys=False), encoding="utf-8")
    out = run(["generate", "stacks.yaml", "--yes"], root)
    stacks = root / "stacks"
    manifest = json.loads((stacks / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["writer"].get("stack_sources") is True, "writer.stack_sources not recorded", out)
    patches = manifest["patches"]
    check(all(set(p["files"]) == {"image", "mask"} for p in patches), "stacked files keys", out)
    names = sorted(f"Images/{p.name}" for p in (stacks / "Images").iterdir())
    check(names == sorted(p["files"]["image"] for p in patches), "stacked Images/ != manifest", out)

    out = run(["generate", "missing.yaml"], root, expect=1)
    check("Traceback" not in out, "a user error printed a traceback", out)

    server.shutdown()
    shutil.rmtree(root.parent, ignore_errors=True)
    print(f"e2e journey passed on {sys.platform} (Python {sys.version.split()[0]})")


if __name__ == "__main__":
    main()
