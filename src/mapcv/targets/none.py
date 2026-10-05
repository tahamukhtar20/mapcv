"""Image-only datasets: no labels, so patches carry no annotation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from mapcv._patching import NullWindow
from mapcv.imagery import RasterMetadata
from mapcv.labels import ClassMap
from mapcv.targets.base import Transform, WindowTarget


class ImageOnlyTarget:
    """A target that annotates nothing."""

    @property
    def class_map(self) -> ClassMap:
        return {}

    def prepare(self, source: RasterMetadata) -> None:
        return None

    def fingerprint(self) -> Optional[Dict[str, Any]]:
        return None

    def window(
        self,
        transform: Transform,
        height: int,
        width: int,
        valid_mask: Optional[npt.NDArray[np.bool_]],
    ) -> WindowTarget:
        return NullWindow()
