"""Positions of a date on the unit circle.

A position such as the day of the year runs from zero up to the length
of the year, and then jumps back to zero. The functions in this module
remove that jump. Each one turns a position into an angle and returns
its sine and cosine, so that the end of a cycle lies next to its start.

There are six cycles: the day of the year, the weekday, the day of the
month, the month, and the week and the day of the ISO 8601 year. The
`cyclic_date` function returns all six. Each position is divided by the
true length of its cycle, so a month of 28 days and a month of 31 days
both run the full circle.

>>> import numpy as np
>>> from urtavla.cycle import cyclic_date
>>> ts = np.array(["2024-12-30"], dtype="datetime64[D]")
>>> cyclic_date(ts).shape
(1, 12)

The year number does not repeat, so it has no cycle of its own. The
`cyclic_year` function therefore measures the year against periods that
the caller chooses, and returns a sine and a cosine for each period.
Long periods keep nearby years close together, while short periods tell
them apart. The `dyadic_period` function builds a set of periods that
double from each to the next.

>>> from urtavla.cycle import cyclic_year, dyadic_period
>>> period = dyadic_period(1, 5)
>>> period
array([ 2.,  4.,  8., 16., 32.])
>>> cyclic_year(ts, period).shape
(1, 10)

The `CyclicYear` class holds a set of periods, for when the periods are
chosen once and reused. An instance is then called with the timestamps
alone.

>>> from urtavla.cycle import CyclicYear
>>> encoder = CyclicYear.from_dyadic(1, 5)
>>> encoder(ts).shape
(1, 10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from urtavla import cardinal, ordinal

if TYPE_CHECKING:
    from typing import Final, Self

    from urtavla.typing import DtND, FloatND


__all__ = (
    "CyclicYear",
    "DyadicPeriod",
    "cyclic_calendar_day_of_month",
    "cyclic_calendar_day_of_year",
    "cyclic_calendar_month",
    "cyclic_date",
    "cyclic_day_of_week",
    "cyclic_iso_day_of_year",
    "cyclic_iso_week_of_year",
    "cyclic_year",
    "dyadic_period",
    "unit_circle",
)


def unit_circle(turn: FloatND) -> FloatND:
    """Map fractions of a turn to points on the unit circle.

    Arguments:
        turn: Fractions of a full turn, where 0 is the start and 1 is
            one whole turn.

    Returns:
        Sine and cosine of each turn, stacked along a new last axis.
    """
    angle = 2.0 * np.pi * turn
    sin = np.sin(angle)
    cos = np.cos(angle)
    return np.stack([sin, cos], axis=-1)


def cyclic_calendar_day_of_year(timestamp: DtND) -> FloatND:
    """Encode the day of the calendar year as a cycle.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Sine and cosine of the position in the calendar year, each from
        -1 to 1.
    """
    index = ordinal.calendar_day_of_year(timestamp)
    total = cardinal.calendar_days_in_year(timestamp)
    return unit_circle(index / total)


def cyclic_day_of_week(timestamp: DtND) -> FloatND:
    """Encode the weekday as a cycle joining Sunday to Monday.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Sine and cosine of the position in the week, each from -1 to 1.
    """
    index = ordinal.day_of_week(timestamp)
    return unit_circle(index / 7)


def cyclic_calendar_day_of_month(timestamp: DtND) -> FloatND:
    """Encode the day of the calendar month as a cycle.

    The position is divided by the length of the month itself, so a day
    is a larger step in a short month than in a long one.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Sine and cosine of the position in the month, each from -1 to 1.
    """
    index = ordinal.calendar_day_of_month(timestamp)
    total = cardinal.calendar_days_in_month(timestamp)
    return unit_circle(index / total)


def cyclic_calendar_month(timestamp: DtND) -> FloatND:
    """Encode the calendar month as a cycle joining December to January.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Sine and cosine of the position in the twelve months, each from
        -1 to 1.
    """
    index = ordinal.calendar_month(timestamp)
    return unit_circle(index / 12)


def cyclic_iso_week_of_year(timestamp: DtND) -> FloatND:
    """Encode the week of the ISO 8601 year as a cycle.

    The position is divided by the number of weeks in the ISO year,
    which is 52 or 53.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Sine and cosine of the position in the ISO year, each from -1 to
        1.
    """
    index = ordinal.iso_week_of_year(timestamp)
    total = cardinal.iso_weeks_in_year(timestamp)
    return unit_circle(index / total)


def cyclic_iso_day_of_year(timestamp: DtND) -> FloatND:
    """Encode the day of the ISO 8601 year as a cycle.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Sine and cosine of the position in the ISO year, each from -1 to
        1.
    """
    index = ordinal.iso_day_of_year(timestamp)
    total = cardinal.iso_days_in_year(timestamp)
    return unit_circle(index / total)


def cyclic_date(timestamp: DtND) -> FloatND:
    """Encode the date as all six cycles.

    Arguments:
        timestamp: Timestamps of any shape and resolution.

    Returns:
        Sine and cosine of each cycle, as twelve columns along a new
        last axis. The cycles come in the order that their functions are
        defined in this module.
    """
    return np.concatenate(
        [
            cyclic_calendar_day_of_year(timestamp),
            cyclic_day_of_week(timestamp),
            cyclic_calendar_day_of_month(timestamp),
            cyclic_calendar_month(timestamp),
            cyclic_iso_week_of_year(timestamp),
            cyclic_iso_day_of_year(timestamp),
        ],
        axis=-1,
    )


def dyadic_period(minimum: int, maximum: int) -> FloatND:
    """Return periods that double from each to the next.

    Arguments:
        minimum: Exponent of the shortest period. The shortest period
            sets how finely neighbouring years are told apart.
        maximum: Exponent of the longest period. The encoding repeats
            after the longest period.

    Returns:
        Periods in years, as powers of two, shortest first.
    """
    return 2.0 ** np.arange(minimum, maximum + 1, dtype=np.float64)


def cyclic_year(timestamp: DtND, period: FloatND) -> FloatND:
    """Encode the calendar year as one cycle per period.

    The year changes in whole steps, which limits the useful periods. A
    period of one year or shorter gives a constant result, and a period
    of two years varies only in its cosine. The whole result repeats
    after the longest period, so years are told apart only within that
    span.

    Arguments:
        timestamp: Timestamps of any shape and resolution.
        period: Period of each cycle, in years.

    Returns:
        Sine and cosine of the year for each period, in the order of the
        periods.
    """
    year = ordinal.calendar_year(timestamp)[..., None]
    array = unit_circle(year / period)
    return array.reshape(*timestamp.shape, -1)


class DyadicPeriod:
    """Hold a fixed set of periods to measure the year against."""

    period: Final[FloatND]
    """Fixed weight array with the yearly periods."""

    def __init__(self, period: FloatND) -> None:
        """Store the periods to encode against.

        Arguments:
            period: Period of each cycle, in years.
        """
        self.period = period

    @classmethod
    def from_dyadic(cls, minimum: int, maximum: int) -> Self:
        """Create an encoder with periods that double.

        Arguments:
            minimum: Exponent of the shortest period.
            maximum: Exponent of the longest period.

        Returns:
            Encoder that holds the periods, shortest first.
        """
        period = dyadic_period(minimum, maximum)
        return cls(period)


class CyclicYear(DyadicPeriod):
    """Encode the calendar year against a fixed set of periods."""

    def __call__(self, timestamp: DtND) -> FloatND:
        """Encode the calendar year as one cycle per period.

        Arguments:
            timestamp: Timestamps of any shape and resolution.

        Returns:
            Sine and cosine of the year for each period, in the order of
            the periods.
        """
        return cyclic_year(timestamp, period=self.period)
