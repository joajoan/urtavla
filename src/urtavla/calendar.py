"""Reference points and units of the calendar.

This module defines the constants that the package computes with: the
epoch, and durations of one day, one month and one year. The
`elapsed_days` function measures a timestamp from the epoch in whole
days.

>>> import numpy as np
>>> from urtavla.calendar import elapsed_days
>>> ts = np.array(["2024-12-30T14:30"], dtype="datetime64[m]")
>>> elapsed_days(ts)
array([20087], dtype='timedelta64[D]')
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from urtavla import types

if TYPE_CHECKING:
    from typing import Final

    from urtavla.typing import DtND, TdND


__all__ = (
    "DAY",
    "EPOCH",
    "EPOCH_YEAR",
    "MONTH",
    "THURSDAY",
    "YEAR",
    "elapsed_days",
)


EPOCH: Final = np.datetime64(0, "D")
"""First day of the UNIX epoch, 1 January 1970."""


EPOCH_YEAR: Final = 1970
"""Year of the UNIX epoch."""


THURSDAY: Final = 3
"""Weekday index of Thursday, the weekday of the epoch."""


DAY: Final = np.timedelta64(1, "D")
"""Duration of one day."""


MONTH: Final = np.timedelta64(1, "M")
"""Duration of one calendar month."""


YEAR: Final = np.timedelta64(1, "Y")
"""Duration of one calendar year."""


def elapsed_days(timestamp: DtND) -> TdND:
    """Return the whole days elapsed since the UNIX epoch.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Days since the epoch, rounded down. The result is a duration,
        not an integer.
    """
    return types.as_days(timestamp - EPOCH)
