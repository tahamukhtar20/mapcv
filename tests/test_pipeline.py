"""End-to-end tests for the source-neutral generation pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pytest

import mapcv

from mapcv.config import LabelsConfig, MapcvConfig
from mapcv.imagery import RasterMetadata
from mapcv.pipeline import run_generate
from mapcv.sampler import PatchMeta
from mapcv.splitter import SplitLists, SplitterConfig
from mapcv.targets import ImageOnlyTarget, SegmentationTarget, WindowTarget, create_target
from mapcv.manifest import (
    Manifest,
    ManifestEntry,
    ManifestMismatchError,
    PatchSummary,
    TargetRecord,
)
from mapcv.writers import FilesWriter, check_compatible, create_writer


class FakeRasterSource:
    """Small chunked raster used to exercise global anchor behavior."""

    def __init__(self) -> None:
        values = np.arange(7 * 5 * 2, dtype=np.float32)
        self.image = values.reshape(7, 5, 2)
        self.valid = np.ones((7, 5), dtype=np.bool_)
        self.windows: List[Tuple[int, int, int, int]] = []
        self.closed = False
        self.metadata = RasterMetadata(
            source_type="eopf_zarr",
            product_id="S2_TEST.zarr",
            width=5,
            height=7,
            bands=["b08", "b04"],
            dtype="float32",
            crs="EPSG:32632",
            transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0),
            chunk_rows=2,
        )

    def read_window(
        self, row_start: int, row_stop: int, col_start: int, col_stop: int
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        self.windows.append((row_start, row_stop, col_start, col_stop))
        return (
            self.image[row_start:row_stop, col_start:col_stop],
            self.valid[row_start:row_stop, col_start:col_stop],
        )

    def close(self) -> None:
        self.closed = True


def _config(tmp_path: Path) -> MapcvConfig:
    return MapcvConfig.model_validate(
        {
            "region": {"west": 9.0, "south": 45.0, "east": 9.1, "north": 45.1},
            "imagery": {
                "type": "eopf_zarr",
                "path": str(tmp_path / "unused.zarr"),
                "bands": ["b08", "b04"],
                "chunk_rows": 2,
            },
            "sampler": {
                "patch_size": 3,
                "stride": 2,
                "edge_strategy": "drop",
            },
            "writer": {"staging_dir": str(tmp_path / "dataset"), "image_format": "npy"},
        }
    )


def test_generate_keeps_global_anchors_across_chunk_seams_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources: List[FakeRasterSource] = []

    def open_source(*args: Any, **kwargs: Any) -> FakeRasterSource:
        source = FakeRasterSource()
        sources.append(source)
        return source

    monkeypatch.setattr("mapcv.pipeline.open_raster_source", open_source)
    config = _config(tmp_path)

    run_generate(config)

    manifest_path = config.writer.staging_dir / "manifest.json"
    manifest = Manifest.load(manifest_path)
    coordinates = [(entry["row"], entry["col"]) for entry in manifest.patches]
    assert coordinates == [(0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2)]
    assert sources[0].windows == [(0, 3, 0, 5), (2, 5, 0, 5), (4, 7, 0, 5)]
    assert sources[0].closed
    assert manifest.version == 3
    assert manifest.task == "segmentation"
    assert manifest.target is None
    source = manifest.source
    assert source.name == "image"
    assert source.source_type == "eopf_zarr"
    assert source.bands == ["b08", "b04"]
    assert source.dtype == "float32"
    assert source.patch_shape == [2, 3, 3]
    assert source.crs == "EPSG:32632"
    assert source.transform == (10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)

    stored = np.load(config.writer.staging_dir / "Images" / "patch_0000000.npy")
    assert stored.shape == (2, 3, 3)
    assert stored.dtype == np.float32

    run_generate(config)

    resumed = Manifest.load(manifest_path)
    assert [(entry["row"], entry["col"]) for entry in resumed.patches] == coordinates
    assert sources[1].windows == []
    assert sources[1].closed


def test_resume_refuses_a_changed_jpg_subsampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: FakeRasterSource())
    config = _config(tmp_path)
    run_generate(config)
    manifest = Manifest.load(config.writer.staging_dir / "manifest.json")
    assert manifest.writer is not None
    assert manifest.writer["jpg_subsampling"] == "4:2:0"

    config.writer = config.writer.model_copy(update={"jpg_subsampling": "4:4:4"})
    with pytest.raises(ManifestMismatchError, match="writer"):
        run_generate(config)


def test_generate_warns_when_labels_miss_the_imagery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = tmp_path / "labels.geojson"
    labels.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
        '"geometry":{"type":"Polygon","coordinates":[[[100,10],[101,10],[101,11],[100,10]]]}}]}'
    )
    monkeypatch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: FakeRasterSource())
    monkeypatch.setattr(
        "mapcv.targets.segmentation.transform_geometry_to_crs", lambda geometry, crs: geometry
    )
    config = _config(tmp_path)
    config.labels = LabelsConfig(path=labels)

    with pytest.warns(UserWarning, match="no label polygon intersects"):
        run_generate(config)


def test_resumed_run_records_the_same_chunk_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources: List[FakeRasterSource] = []

    def open_source(*args: Any, **kwargs: Any) -> FakeRasterSource:
        sources.append(FakeRasterSource())
        return sources[-1]

    monkeypatch.setattr("mapcv.pipeline.open_raster_source", open_source)
    config = _config(tmp_path)
    run_generate(config)
    manifest_path = config.writer.staging_dir / "manifest.json"
    complete = Manifest.load(manifest_path)
    assert [entry["chunk"] for entry in complete.patches] == [0, 0, 1, 1, 2, 2]

    # Simulate an interruption after the first chunk.
    partial = Manifest.load(manifest_path)
    partial.patches = partial.patches[:2]
    partial.save(manifest_path)
    run_generate(config)

    assert sources[1].windows == [(2, 5, 0, 5), (4, 7, 0, 5)]
    assert Manifest.load(manifest_path).patches == complete.patches


@pytest.mark.parametrize(("max_empty_ratio", "warns"), [(1.0, True), (0.5, False)])
def test_generate_warns_when_failed_tiles_stay_in_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_empty_ratio: float, warns: bool
) -> None:
    def open_source(*args: Any, **kwargs: Any) -> FakeRasterSource:
        source = FakeRasterSource()
        source.tiles_requested = 10  # type: ignore[attr-defined]
        source.tiles_failed = 1  # type: ignore[attr-defined]
        return source

    monkeypatch.setattr("mapcv.pipeline.open_raster_source", open_source)
    config = _config(tmp_path)
    config.sampler.max_empty_ratio = max_empty_ratio

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_generate(config)

    assert any("filled with black" in str(w.message) for w in caught) is warns


@pytest.mark.parametrize("all_touched", [False, True])
def test_windowed_label_selection_matches_full_rasterization(all_touched: bool) -> None:
    from shapely.geometry import box as make_box

    from mapcv.targets.segmentation import _geometries_in_window, _label_bounds
    from mapcv.rasterizer import rasterize

    rng = np.random.default_rng(7)
    geometries = []
    for _ in range(400):
        x, y = rng.uniform(0, 1000, 2)
        w, h = rng.uniform(0.3, 40, 2)
        # Overlapping boxes with mixed classes: order decides the burned value.
        geometries.append((make_box(x, y, x + w, y + h), int(rng.integers(1, 4))))
    bounds = _label_bounds(geometries)
    for _ in range(25):
        col, row = rng.integers(0, 900, 2)
        height, width = rng.integers(1, 120, 2)
        transform = (1.0, 0.0, float(col), 0.0, -1.0, 1000.0 - float(row))
        nearby = _geometries_in_window(geometries, bounds, transform, int(height), int(width))
        assert len(nearby) < len(geometries)
        expected = rasterize(geometries, (int(height), int(width)), transform, all_touched)
        actual = rasterize(nearby, (int(height), int(width)), transform, all_touched)
        np.testing.assert_array_equal(actual, expected)


def _labeled_config(tmp_path: Path, labels_json: str, **labels: Any) -> MapcvConfig:
    path = tmp_path / "labels.geojson"
    path.write_text(labels_json)
    config = _config(tmp_path)
    config.labels = LabelsConfig(path=path, **labels)
    config.sampler.edge_strategy = "pad"
    return config


# A square covering the whole fake raster (transform is identity in these tests).
_COVER_ALL = (
    '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"kind":"7"},'
    '"geometry":{"type":"Polygon","coordinates":[[[400000,4000000],[600000,4000000],'
    "[600000,6000000],[400000,6000000],[400000,4000000]]]}}]}"
)


def test_generated_masks_mark_padding_with_the_ignore_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    monkeypatch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: FakeRasterSource())
    monkeypatch.setattr(
        "mapcv.targets.segmentation.transform_geometry_to_crs", lambda geometry, crs: geometry
    )
    config = _labeled_config(tmp_path, _COVER_ALL)
    run_generate(config)

    manifest = Manifest.load(config.writer.staging_dir / "manifest.json")
    assert manifest.target is not None and manifest.target.ignore_index == 255
    assert manifest.ignore_index == 255
    padded = [entry for entry in manifest.patches if entry["padded"]]
    assert padded
    for entry in padded:
        mask = np.asarray(Image.open(config.writer.staging_dir / entry["files"]["mask"]))
        assert set(np.unique(mask)) == {1, 255}  # the class inside, ignore in the padding
        assert set(entry["summary"]["class_pixels"]) == {"1", "255"}


def test_a_class_on_the_ignore_value_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: FakeRasterSource())
    monkeypatch.setattr(
        "mapcv.targets.segmentation.transform_geometry_to_crs", lambda geometry, crs: geometry
    )
    config = _labeled_config(tmp_path, _COVER_ALL.replace('"7"', '"255"'), label_field="kind")
    with pytest.raises(ValueError, match="labels.ignore_index"):
        run_generate(config)


def test_global_random_anchors_are_distinct_and_warn_when_capped() -> None:
    from mapcv.pipeline import _global_anchors
    from mapcv.sampler import SamplerConfig

    config = SamplerConfig(
        patch_size=4, mode="random", random_count=500, random_seed=1, edge_strategy="drop"
    )
    with pytest.warns(UserWarning, match=r"only 49 distinct patch position"):
        anchors = _global_anchors(10, 10, config)
    assert len(anchors) == len(set(anchors)) == 49
    with pytest.warns(UserWarning):
        assert anchors == _global_anchors(10, 10, config)

    enough = config.model_copy(update={"random_count": 20})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(_global_anchors(10, 10, enough)) == 20


# ---------------------------------------------------------------------------
# The pipeline is task-agnostic: any Target and Writer plug in.
# ---------------------------------------------------------------------------

Center = Tuple[int, int]
Transform = Tuple[float, float, float, float, float, float]


class CenterWindow:
    """Annotates each patch with its centre pixel in window coordinates."""

    def __init__(self, owner: "CenterTarget", transform: Transform, shape: Tuple[int, int]):
        self.owner = owner
        self.transform = transform
        self.shape = shape

    def annotate(
        self,
        row: int,
        col: int,
        patch_size: int,
        pad_mode: str,
        valid_patch: Optional[npt.NDArray[np.bool_]],
    ) -> Center:
        return (row + patch_size // 2, col + patch_size // 2)

    def accepts(self, annotation: Center, min_label_ratio: float) -> bool:
        return annotation != self.owner.reject

    def collate(self, annotations: Sequence[Center], patch_size: int) -> List[Center]:
        return list(annotations)


class CenterTarget:
    def __init__(self, reject: Optional[Center] = None) -> None:
        self.reject = reject
        self.prepared_with: List[RasterMetadata] = []
        self.windows: List[CenterWindow] = []

    @property
    def type(self) -> Optional[str]:
        return "centers"

    @property
    def class_map(self) -> Dict[str, int]:
        return {"center": 1}

    def prepare(self, source: RasterMetadata) -> None:
        self.prepared_with.append(source)

    def record(self) -> Optional[TargetRecord]:
        return TargetRecord(type="centers", class_map=self.class_map, options={"radius": 0})

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: Optional[npt.NDArray[np.bool_]],
    ) -> WindowTarget:
        assert valid_mask is not None and valid_mask.shape == (height, width)
        window = CenterWindow(self, transform, (height, width))
        self.windows.append(window)
        return window


class RecordingWriter:
    def __init__(self) -> None:
        self.calls: List[Tuple[int, List[Center], List[Tuple[int, int]], Tuple[int, ...]]] = []
        self.finalized: List[Tuple[int, Optional[SplitLists]]] = []
        self.supported: Tuple[Optional[str], ...] = ("centers",)

    @property
    def layout(self) -> str:
        return "recording"

    def supports(self, target_type: Optional[str]) -> bool:
        return target_type in self.supported

    def fingerprint(self) -> Dict[str, Any]:
        return {"layout": "recording"}

    def patch_shape(self, source: RasterMetadata, patch_size: int) -> List[int]:
        return [len(source.bands), patch_size, patch_size]

    def write(
        self,
        images: npt.NDArray[Any],
        annotations: List[Center],
        metadata: List[PatchMeta],
        manifest: Manifest,
        chunk_index: int,
    ) -> None:
        self.calls.append(
            (
                chunk_index,
                annotations,
                [(m["row"], m["col"]) for m in metadata],
                tuple(images.shape),
            )
        )
        for index, patch in enumerate(metadata):
            manifest.patches.append(
                ManifestEntry(
                    row=patch["row"],
                    col=patch["col"],
                    padded=patch["padded"],
                    chunk=chunk_index,
                    files={"image": f"Records/{len(manifest.patches)}.bin"},
                    summary=PatchSummary(empty_ratio=patch.get("empty_ratio", 0.0)),
                )
            )

    def finalize(self, manifest: Manifest, split_lists: Optional[SplitLists]) -> None:
        self.finalized.append((len(manifest.patches), split_lists))


def _plug(
    monkeypatch: pytest.MonkeyPatch, target: CenterTarget, writer: RecordingWriter
) -> List[FakeRasterSource]:
    sources: List[FakeRasterSource] = []

    def open_source(*args: Any, **kwargs: Any) -> FakeRasterSource:
        sources.append(FakeRasterSource())
        return sources[-1]

    monkeypatch.setattr("mapcv.pipeline.open_raster_source", open_source)
    monkeypatch.setattr("mapcv.pipeline.create_target", lambda config: target)
    monkeypatch.setattr("mapcv.pipeline.create_writer", lambda config, target=None: writer)
    return sources


def test_pipeline_drives_any_target_and_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, writer = CenterTarget(), RecordingWriter()
    sources = _plug(monkeypatch, target, writer)
    config = _config(tmp_path)

    result = run_generate(config)

    assert len(target.prepared_with) == 1 and target.prepared_with[0].product_id == "S2_TEST.zarr"
    # One window per chunk, offset to the chunk origin: rows 0, 2 and 4 of the raster.
    assert [w.transform[5] for w in target.windows] == [5000000.0, 4999980.0, 4999960.0]
    assert [w.shape for w in target.windows] == [(3, 5), (3, 5), (3, 5)]
    assert sources[0].windows == [(0, 3, 0, 5), (2, 5, 0, 5), (4, 7, 0, 5)]
    # The writer receives each chunk's collated annotations, patch metadata and images.
    assert [call[0] for call in writer.calls] == [0, 1, 2]
    # Annotations are in window coordinates, so every chunk (3 rows high) looks the same.
    assert [call[1] for call in writer.calls] == [[(1, 1), (1, 3)]] * 3
    assert [call[2] for call in writer.calls] == [
        [(0, 0), (0, 2)],
        [(2, 0), (2, 2)],
        [(4, 0), (4, 2)],
    ]
    assert all(call[3] == (2, 3, 3, 2) for call in writer.calls)
    # The manifest takes its descriptive blocks from the target and the writer.
    manifest = Manifest.load(config.writer.staging_dir / "manifest.json")
    assert manifest.class_map == {"center": 1}
    assert manifest.target == TargetRecord(
        type="centers", class_map={"center": 1}, options={"radius": 0}
    )
    assert manifest.writer == {"layout": "recording"}
    assert manifest.source.patch_shape == [2, 3, 3]
    assert len(manifest.patches) == result.new_patches == 6
    assert writer.finalized == [(6, None)]


def test_a_target_can_reject_patches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target, writer = CenterTarget(reject=(1, 3)), RecordingWriter()
    _plug(monkeypatch, target, writer)

    result = run_generate(_config(tmp_path))

    assert result.new_patches == 3
    assert [call[2] for call in writer.calls] == [[(0, 0)], [(2, 0)], [(4, 0)]]


@pytest.mark.filterwarnings("ignore:Patches overlap")
def test_finalize_runs_last_with_the_split_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, writer = CenterTarget(), RecordingWriter()
    _plug(monkeypatch, target, writer)
    config = _config(tmp_path)
    config.split = SplitterConfig(strategy="random", test_ratio=0.34, val_ratio=0.0)

    result = run_generate(config)

    assert writer.finalized[0][0] == 6
    lists = writer.finalized[0][1]
    assert lists is not None
    splits = config.writer.staging_dir / "splits"
    for name in ("train", "val", "test"):
        assert getattr(lists, name) == (splits / f"{name}.txt").read_text().splitlines()
    assert result.split_counts is not None
    assert result.split_counts["train"] == len(lists.train)
    assert result.split_counts["test"] == len(lists.test)
    assert sorted(lists.train + lists.val + lists.test) == sorted(
        f"{index}.bin" for index in range(6)
    )


def test_resume_skips_writes_but_still_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, writer = CenterTarget(), RecordingWriter()
    sources = _plug(monkeypatch, target, writer)
    config = _config(tmp_path)
    run_generate(config)
    writes = len(writer.calls)

    run_generate(config)

    assert len(writer.calls) == writes
    assert sources[1].windows == []
    assert [count for count, _ in writer.finalized] == [6, 6]


def test_default_factories_pick_the_segmentation_pieces(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert isinstance(create_target(config), ImageOnlyTarget)
    assert isinstance(create_writer(config.writer), FilesWriter)
    config.labels = LabelsConfig(path=tmp_path / "labels.geojson")
    assert isinstance(create_target(config), SegmentationTarget)


def test_targets_need_prepare_before_use(tmp_path: Path) -> None:
    target = SegmentationTarget(LabelsConfig(path=tmp_path / "labels.geojson"))
    with pytest.raises(RuntimeError, match="prepare"):
        _ = target.class_map
    with pytest.raises(RuntimeError, match="prepare"):
        target.record()
    assert target.type == "segmentation"
    assert ImageOnlyTarget().class_map == {}
    assert ImageOnlyTarget().record() is None
    assert ImageOnlyTarget().type is None


def test_files_writer_describes_its_layout(tmp_path: Path) -> None:
    source = FakeRasterSource().metadata
    npy = FilesWriter(_config(tmp_path).writer)
    assert npy.patch_shape(source, 3) == [2, 3, 3]
    assert npy.fingerprint() == {
        "layout": "files",
        "image_format": "npy",
        "jpg_quality": 95,
        "jpg_subsampling": "4:2:0",
        "mask_format": "png",
    }
    assert npy.layout == "files"
    assert npy.supports("segmentation") and npy.supports(None)
    assert not npy.supports("detection")

    png = FilesWriter(_config(tmp_path).writer.model_copy(update={"image_format": "png"}))
    assert png.patch_shape(source, 3) == [3, 3, 3]


def test_files_writer_only_takes_masks(tmp_path: Path) -> None:
    writer = FilesWriter(_config(tmp_path).writer)
    images = np.zeros((1, 3, 3, 2), dtype=np.float32)
    meta = [PatchMeta(row=0, col=0, padded=False, empty_ratio=0.0)]
    with pytest.raises(TypeError, match="masks"):
        writer.write(images, [(1, 1)], meta, Manifest(), 0)  # type: ignore[arg-type]


def test_an_incompatible_writer_fails_before_reading_imagery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, writer = CenterTarget(), RecordingWriter()
    writer.supported = ("segmentation",)
    sources = _plug(monkeypatch, target, writer)

    with pytest.raises(ValueError, match="'recording' writer layout cannot write centers"):
        run_generate(_config(tmp_path))

    assert sources == [] and target.prepared_with == []
    assert not (tmp_path / "dataset" / "manifest.json").exists()


def test_check_compatible_accepts_the_default_pieces(tmp_path: Path) -> None:
    config = _config(tmp_path)
    writer = create_writer(config.writer)
    check_compatible(create_target(config), writer)
    config.labels = LabelsConfig(path=tmp_path / "labels.geojson")
    check_compatible(create_target(config), writer)


def test_generate_records_the_task_and_the_patch_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mapcv.pipeline.open_raster_source", lambda *a, **k: FakeRasterSource())
    monkeypatch.setattr(
        "mapcv.targets.segmentation.transform_geometry_to_crs", lambda geometry, crs: geometry
    )
    config = _labeled_config(tmp_path, _COVER_ALL)
    manifest = run_generate(config).manifest

    assert manifest.task == "segmentation"
    assert manifest.mapcv_version == mapcv.__version__
    staging = config.writer.staging_dir
    for index, entry in enumerate(manifest.patches):
        assert entry["files"] == {
            "image": f"Images/patch_{index:07d}.npy",
            "mask": f"Masks/patch_{index:07d}.png",
        }
        assert all((staging / path).is_file() for path in entry["files"].values())
