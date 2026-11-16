from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from urtavla import cardinal

if TYPE_CHECKING:
    from urtavla.typing import DtND


def test_calendar_days_in_month(timestamp: DtND) -> None:
    actual = cardinal.calendar_days_in_month(timestamp)
    np.testing.assert_array_equal(actual, [31, 29, 31, 31])


def test_calendar_days_in_year(timestamp: DtND) -> None:
    actual = cardinal.calendar_days_in_year(timestamp)
    np.testing.assert_array_equal(actual, [366, 366, 365, 365])


def test_iso_weeks_in_year(timestamp: DtND) -> None:
    actual = cardinal.iso_weeks_in_year(timestamp)
    np.testing.assert_array_equal(actual, [52, 53, 53, 53])
