"""Change detection datasets: ``A/`` (before), ``B/`` (after) and ``label/`` (change masks)."""

from __future__ import annotations

from mapcv.writer import WriterConfig
from mapcv.writers.files import FilesWriter

# LEVIR-CD's folder names, which change-detection loaders expect.
BEFORE_DIR = "A"
AFTER_DIR = "B"
LABEL_DIR = "label"


class ChangeWriter(FilesWriter):
    """The before and after patches of a change dataset, with one change mask each.

    The layout of LEVIR-CD and the loaders built for it: ``A/patch_*.<ext>`` holds the
    before image (the first source), ``B/`` the after image under the same name and
    ``label/`` the change mask. Splits are the usual ``splits/<split>.txt`` lists of
    those names, so the folders stay put when the dataset is re-split.
    """

    TARGET_TYPES: frozenset[str | None] = frozenset({"change"})

    def __init__(self, config: WriterConfig, sources: list[str]) -> None:
        if len(sources) != 2:
            raise ValueError("a change dataset has two imagery sources: before and after")
        super().__init__(config, sources)
        self._folders = {sources[0]: BEFORE_DIR, sources[1]: AFTER_DIR}

    @property
    def layout(self) -> str:
        return "change"

    def _images_dir(self, name: str) -> str:
        return self._folders[name]

    @property
    def _masks_dir(self) -> str:
        return LABEL_DIR
