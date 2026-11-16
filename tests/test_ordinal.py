from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from urtavla import ordinal

if TYPE_CHECKING:
    from urtavla.typing import DtND


def test_calendar_year(timestamp: DtND) -> None:
    actual = ordinal.calendar_year(timestamp)
    np.testing.assert_array_equal(actual, [1968, 2020, 2021, 2021])


def test_calendar_month(timestamp: DtND) -> None:
    actual = ordinal.calendar_month(timestamp)
    np.testing.assert_array_equal(actual, [11, 1, 0, 0])


def test_calendar_day_of_year(timestamp: DtND) -> None:
    actual = ordinal.calendar_day_of_year(timestamp)
    np.testing.assert_array_equal(actual, [365, 59, 0, 2])


def test_calendar_day_of_month(timestamp: DtND) -> None:
    actual = ordinal.calendar_day_of_month(timestamp)
    np.testing.assert_array_equal(actual, [30, 28, 0, 2])


def test_day_of_week(timestamp: DtND) -> None:
    actual = ordinal.day_of_week(timestamp)
    np.testing.assert_array_equal(actual, [1, 5, 4, 6])


def test_iso_year(timestamp: DtND) -> None:
    actual = ordinal.iso_year(timestamp)
    np.testing.assert_array_equal(actual, [1969, 2020, 2020, 2020])


def test_iso_week_of_year(timestamp: DtND) -> None:
    actual = ordinal.iso_week_of_year(timestamp)
    np.testing.assert_array_equal(actual, [0, 8, 52, 52])


def test_iso_day_of_year(timestamp: DtND) -> None:
    actual = ordinal.iso_day_of_year(timestamp)
    np.testing.assert_array_equal(actual, [1, 61, 368, 370])
