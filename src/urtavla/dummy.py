"""Flags that answer yes-or-no questions about a date.

A flag is true where the answer is yes, and false elsewhere. There are
five flags. They tell whether a date falls in a leap year, on the leap
day or on a weekend, and whether it is the first or the last day of a
month. Each flag has a function of its own, and the `dummy_date`
function returns all five.

>>> import numpy as np
>>> from urtavla.dummy import dummy_date
>>> ts = np.array(["2024-02-29"], dtype="datetime64[D]")
>>> dummy_date(ts)
array([[ True,  True, False, False,  True]])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from urtavla import cardinal, ordinal

if TYPE_CHECKING:
    from typing import Final

    from urtavla.typing import BoolND, DtND


__all__ = (
    "dummy_date",
    "dummy_leap_day",
    "dummy_leap_year",
    "dummy_month_end",
    "dummy_month_start",
    "dummy_weekend",
)


DAYS_IN_LEAP_YEAR: Final = 366
"""Length that only a leap year has."""

DAYS_IN_FEBRUARY: Final = 29
"""Length that only February in a leap year has."""


def dummy_leap_year(timestamp: DtND) -> BoolND:
    """Flag every day of a leap year.

    A leap year is recognised by its length, not by its year number.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        True on every day of a year that has 366 days.
    """
    return cardinal.calendar_days_in_year(timestamp) == DAYS_IN_LEAP_YEAR


def dummy_leap_day(timestamp: DtND) -> BoolND:
    """Flag 29 February.

    A leap year is recognised by the length of its February, not by its
    year number.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        True on 29 February, false on every other day.
    """
    day = ordinal.calendar_day_of_month(timestamp)
    total = cardinal.calendar_days_in_month(timestamp)
    return (day == DAYS_IN_FEBRUARY - 1) & (total == DAYS_IN_FEBRUARY)


def dummy_weekend(timestamp: DtND, *, threshold: int = 5) -> BoolND:
    """Flag Saturdays and Sundays.

    Arguments:
        timestamp: Timestamps of any shape and resolution.
        threshold: First weekday index to flag. The default of 5 flags
            Saturday and Sunday.

    Returns:
        True on a weekday index at or above the threshold.
    """
    return ordinal.day_of_week(timestamp) >= threshold


def dummy_month_start(timestamp: DtND) -> BoolND:
    """Flag the first day of the calendar month.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        True on the first day of a month.
    """
    return ordinal.calendar_day_of_month(timestamp) < 1


def dummy_month_end(timestamp: DtND) -> BoolND:
    """Flag the last day of the calendar month.

    The day is compared with the length of its own month, so the flag
    also holds in February.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        True on the last day of a month.
    """
    day = ordinal.calendar_day_of_month(timestamp)
    total = cardinal.calendar_days_in_month(timestamp)
    return day >= total - 1


def dummy_date(timestamp: DtND) -> BoolND:
    """Stack the five flags along a new last axis.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        One column per flag, in the order that the functions are defined
        in this module.
    """
    return np.stack(
        [
            dummy_leap_year(timestamp),
            dummy_leap_day(timestamp),
            dummy_weekend(timestamp),
            dummy_month_start(timestamp),
            dummy_month_end(timestamp),
        ],
        axis=-1,
    )
