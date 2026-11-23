"""Every feature of a date as one matrix.

The `cycle` module gives the cycles of a date, and the `dummy` module
gives its flags. The `encode_date` function joins the two along the last
axis, so that a model gets the whole date as one row of numbers. The
cycles come first. The flags follow as zeros and ones, because a matrix
has a single type.

>>> import numpy as np
>>> from urtavla.cycle import dyadic_period
>>> from urtavla.encode import encode_date
>>> ts = np.array(["2024-02-29"], dtype="datetime64[D]")
>>> period = dyadic_period(1, 5)
>>> encode_date(ts, period).shape
(1, 27)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from urtavla import cycle, dummy

if TYPE_CHECKING:
    from urtavla.typing import DtND, FloatND


__all__ = ("encode_date",)


def encode_date(timestamp: DtND, period: FloatND) -> FloatND:
    """Encode the date as its cycles and its flags.

    Arguments:
        timestamp: Timestamps of any shape and resolution.
        period: Period of each cycle of the year, in years.

    Returns:
        Columns of `cyclic_date`, then those of `dummy_date`, along a
        new last axis. A flag is 1.0 where it holds and 0.0 where it
        does not.
    """
    cyclic = cycle.cyclic_date(timestamp, period)
    flag = dummy.dummy_date(timestamp)
    return np.concatenate([cyclic, flag], axis=-1)
