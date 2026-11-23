"""Positions of a timestamp in the calendar.

Each function in this module returns one position as an integer. Months,
weeks and days are counted from zero, so January is month 0 and Monday
is weekday 0. Years keep their usual number.

The positions exist in two calendar systems. The calendar year starts on
1 January, while the ISO 8601 year starts on the Monday of its first
week. The two therefore disagree about the year of some dates near New
Year, as 30 December 2024 shows below.

>>> import numpy as np
>>> from urtavla import ordinal
>>> ts = np.array(["2024-12-30"], dtype="datetime64[D]")
>>> ordinal.calendar_year(ts)
array([2024])
>>> ordinal.iso_year(ts)
array([2025])
>>> ordinal.iso_week_of_year(ts)
array([0])
>>> ordinal.day_of_week(ts)
array([0])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from urtavla import calendar, types

if TYPE_CHECKING:
    from urtavla.typing import DtND, IntND


__all__ = (
    "calendar_day_of_month",
    "calendar_day_of_year",
    "calendar_month",
    "calendar_year",
    "day_of_week",
    "iso_day_of_year",
    "iso_week_of_year",
    "iso_year",
)


def calendar_year(timestamp: DtND) -> IntND:
    """Return the calendar year.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Year numbers, such as 2024.
    """
    timestamp = types.as_year(timestamp)
    return types.as_int(timestamp) + calendar.EPOCH_YEAR


def calendar_month(timestamp: DtND) -> IntND:
    """Return the zero-based month of the calendar year.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Month index from 0 to 11, where 0 is January.
    """
    array = types.as_month(timestamp)
    return types.as_int(array) % 12


def calendar_day_of_year(timestamp: DtND) -> IntND:
    """Return the zero-based day of the calendar year.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Days since 1 January, from 0 to 365.
    """
    array = types.as_year(timestamp)
    array = types.as_day(array)
    delta = types.as_days(timestamp - array)
    return types.as_int(delta)


def calendar_day_of_month(timestamp: DtND) -> IntND:
    """Return the zero-based day of the calendar month.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Days since the first of the month, from 0 to 30.
    """
    array = types.as_month(timestamp)
    array = types.as_day(array)
    delta = types.as_days(timestamp - array)
    return types.as_int(delta)


def day_of_week(timestamp: DtND) -> IntND:
    """Return the zero-based weekday starting on Monday.

    The calendar year and the ISO year number the weekdays in the same
    way, so one function serves both.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Weekday index from 0 to 6, where 0 is Monday.
    """
    weekday = calendar.THURSDAY * calendar.DAY
    delta = calendar.elapsed_days(timestamp) + weekday
    return types.as_int(delta) % 7


def iso_year(timestamp: DtND) -> IntND:
    """Return the ISO 8601 year.

    The ISO year differs from the calendar year for some dates near New
    Year. The last days of December can belong to the next ISO year, and
    the first days of January to the previous one.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Year numbers, such as 2025.
    """
    timestamp = calendar.iso_thursday(timestamp)
    return calendar_year(timestamp)


def iso_week_of_year(timestamp: DtND) -> IntND:
    """Return the zero-based week of the ISO 8601 year.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Week index from 0 to 52.
    """
    thursday = calendar.iso_thursday(timestamp)
    year = types.as_year(thursday)
    return types.as_int(thursday - year) // 7


def iso_day_of_year(timestamp: DtND) -> IntND:
    """Return the zero-based day of the ISO 8601 year.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Days since the first Monday of the ISO year, from 0 to 370.
    """
    return iso_week_of_year(timestamp) * 7 + day_of_week(timestamp)
