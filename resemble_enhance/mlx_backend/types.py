from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import mlx.core as mx
import numpy as np

Array: TypeAlias = mx.array
NDArray: TypeAlias = np.ndarray
PathLike: TypeAlias = str | Path
