"""Reference points and units of the calendar.

This module defines the constants that the package computes with: the
epoch, and durations of one day, one month and one year. It also
provides the two reference points that `ordinal` and `cardinal` build
on. The `elapsed_days` function measures a timestamp from the epoch in
whole days. The `iso_thursday` function finds the Thursday of a
timestamp's ISO 8601 week.

>>> import numpy as np
>>> from urtavla.calendar import elapsed_days, iso_thursday
>>> ts = np.array(["2024-12-30T14:30"], dtype="datetime64[m]")
>>> elapsed_days(ts)
array([20087], dtype='timedelta64[D]')
>>> iso_thursday(ts)
array(['2025-01-02'], dtype='datetime64[D]')
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
    "iso_thursday",
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


def iso_thursday(timestamp: DtND) -> DtND:
    """Return the Thursday of each timestamp's ISO 8601 week.

    ISO 8601 assigns a week to the year that holds its Thursday. Every
    day of a week shares that Thursday, so it identifies both the week
    and its ISO year. The `ordinal` and `cardinal` modules compute from
    it.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Date of the Thursday, at day resolution.
    """
    timestamp = types.as_day(timestamp)
    delta = types.as_int(timestamp - EPOCH)
    offset = (delta + THURSDAY) % 7
    # The Thursday is a day count from the epoch, and casting it back
    # to a date reinterprets it as such.
    return types.as_day(delta - offset + THURSDAY)
