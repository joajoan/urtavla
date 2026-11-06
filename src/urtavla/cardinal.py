"""Lengths of the year and month that hold a timestamp.

Each function in this module returns one length as an integer: the days
in the month, the days in the year, or the weeks in the year. Every
length is measured from the timestamp itself, by stepping to the start
of the next month or year, so the module holds no leap-year rule.

The year has a length in two calendar systems. A calendar year has 365
or 366 days. An ISO 8601 year is made of whole weeks, so it has 364 or
371 days. The two systems therefore never agree on the length of a year
in days.

>>> import numpy as np
>>> from urtavla import cardinal
>>> ts = np.array(["2026-11-09"], dtype="datetime64[D]")
>>> cardinal.calendar_days_in_month(ts)
array([30])
>>> cardinal.calendar_days_in_year(ts)
array([365])
>>> cardinal.iso_weeks_in_year(ts)
array([53])
>>> cardinal.iso_days_in_year(ts)
array([371])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from urtavla import calendar, types

if TYPE_CHECKING:
    from urtavla.typing import DtND, IntND


__all__ = (
    "calendar_days_in_month",
    "calendar_days_in_year",
    "iso_days_in_year",
    "iso_weeks_in_year",
)


def calendar_days_in_year(timestamp: DtND) -> IntND:
    """Return the number of days in the calendar year.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        365, or 366 in a leap year.
    """
    timestamp = types.as_year(timestamp)
    delta = types.as_day(timestamp + calendar.YEAR) - types.as_day(timestamp)
    return types.as_int(delta)


def calendar_days_in_month(timestamp: DtND) -> IntND:
    """Return the number of days in the calendar month.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Day count from 28 to 31.
    """
    timestamp = types.as_month(timestamp)
    delta = types.as_day(timestamp + calendar.MONTH) - types.as_day(timestamp)
    return types.as_int(delta)


def iso_weeks_in_year(timestamp: DtND) -> IntND:
    """Return the number of weeks in the ISO 8601 year.

    The weeks are counted as the Thursdays of the year. The last
    Thursday is found from 28 December, which always falls in the final
    week. A year has 53 weeks when 1 January is a Thursday, or a
    Wednesday in a leap year.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        52 or 53.
    """
    current = calendar.iso_thursday(timestamp)
    current = types.as_year(current)
    future = types.as_day(current + calendar.YEAR)
    future = calendar.iso_thursday(future - 4 * calendar.DAY)
    return types.as_int(future - current) // 7 + 1


def iso_days_in_year(timestamp: DtND) -> IntND:
    """Return the number of days in the ISO 8601 year.

    An ISO year is made of whole weeks, so its length is a multiple of
    seven. It therefore never equals the 365 or 366 days of the calendar
    year.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        364 or 371.
    """
    return iso_weeks_in_year(timestamp) * 7
