"""Casts of timestamps and durations between resolutions.

A NumPy timestamp carries a resolution, such as minutes, days or months.
The functions in this module cast a timestamp to another resolution. A
cast to a coarser resolution rounds down, so every minute of November
becomes November. A cast to a finer resolution gives the start of the
period, so November becomes 1 November.

A timestamp can also be read as an integer. The integer counts the units
of the resolution since the epoch, 1 January 1970.

>>> import numpy as np
>>> from urtavla import types
>>> ts = np.array(["2026-11-09T14:30"], dtype="datetime64[m]")
>>> ts = types.as_month(ts)
>>> ts
array(['2026-11'], dtype='datetime64[M]')
>>> ts = types.as_day(ts)
>>> ts
array(['2026-11-01'], dtype='datetime64[D]')
>>> types.as_int(ts)
array([20758])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from urtavla.typing import DtND, IntND, TdND


__all__ = (
    "as_day",
    "as_days",
    "as_int",
    "as_month",
    "as_year",
)


def as_day(timestamp: DtND | IntND) -> DtND:
    """Round timestamps down to day resolution.

    Arguments:
        timestamp: Timestamps of any shape and resolution, or integer
            counts of days since the epoch.

    Returns:
        The timestamps at day resolution.
    """
    return timestamp.astype("datetime64[D]")


def as_month(timestamp: DtND | IntND) -> DtND:
    """Round timestamps down to month resolution.

    Arguments:
        timestamp: Timestamps of any shape and resolution, or integer
            counts of months since the epoch.

    Returns:
        The timestamps at month resolution.
    """
    return timestamp.astype("datetime64[M]")


def as_year(timestamp: DtND | IntND) -> DtND:
    """Round timestamps down to year resolution.

    Arguments:
        timestamp: Timestamps of any shape and resolution, or integer
            counts of years since the epoch.

    Returns:
        The timestamps at year resolution.
    """
    return timestamp.astype("datetime64[Y]")


def as_int(value: DtND | TdND) -> IntND:
    """Read timestamps or durations as integers.

    Arguments:
        value: Timestamps or durations of any shape and resolution.

    Returns:
        Count of the units that each value is measured in. For a
        timestamp, the count starts at the epoch.
    """
    return value.astype("int64")


def as_days(delta: TdND) -> TdND:
    """Round durations down to whole days.

    Arguments:
        delta: Durations of any shape and resolution.

    Returns:
        The durations in whole days. A negative duration rounds away
        from zero, so minus one hour becomes minus one day.
    """
    return delta.astype("timedelta64[D]")
