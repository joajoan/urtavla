"""Aliases for the array types used throughout the package.

Each alias names a NumPy array by what it holds, so that a function
signature states plainly what the function expects and returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


__all__ = ("BoolND", "DtND", "FloatND", "IntND", "TdND")


type DtND = NDArray[np.datetime64]
"""Array of timestamps of any shape and resolution."""

type TdND = NDArray[np.timedelta64]
"""Array of durations of any shape and resolution."""

type IntND = NDArray[np.int64]
"""Array of 64-bit integers of any shape."""

type BoolND = NDArray[np.bool]
"""Array of booleans of any shape."""

type FloatND = NDArray[np.float64]
"""Array of 64-bit floats of any shape."""
