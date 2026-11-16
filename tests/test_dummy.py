from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from urtavla import dummy

if TYPE_CHECKING:
    from urtavla.typing import BoolND, DtND


@pytest.fixture
def leap_day() -> BoolND:
    return np.array([False, True, False, False])


def test_dummy_leap_day(timestamp: DtND, leap_day: BoolND) -> None:
    actual = dummy.dummy_leap_day(timestamp)
    np.testing.assert_array_equal(actual, leap_day)


@pytest.fixture
def weekend() -> BoolND:
    return np.array([False, True, False, True])


def test_dummy_weekend(timestamp: DtND, weekend: BoolND) -> None:
    actual = dummy.dummy_weekend(timestamp)
    np.testing.assert_array_equal(actual, weekend)


@pytest.fixture
def month_start() -> BoolND:
    return np.array([False, False, True, False])


def test_dummy_month_start(timestamp: DtND, month_start: BoolND) -> None:
    actual = dummy.dummy_month_start(timestamp)
    np.testing.assert_array_equal(actual, month_start)


@pytest.fixture
def month_end() -> BoolND:
    return np.array([True, True, False, False])


def test_dummy_month_end(timestamp: DtND, month_end: BoolND) -> None:
    actual = dummy.dummy_month_end(timestamp)
    np.testing.assert_array_equal(actual, month_end)


@pytest.fixture
def dummy_array(
    leap_day: BoolND,
    weekend: BoolND,
    month_start: BoolND,
    month_end: BoolND,
) -> BoolND:
    return np.stack([leap_day, weekend, month_start, month_end], axis=-1)


def test_dummy_date(
    timestamp: DtND,
    dummy_array: BoolND,
) -> None:
    actual = dummy.dummy_date(timestamp)
    np.testing.assert_array_equal(actual, dummy_array)


def test_dummy_date_keeps_leading_shape(timestamp: DtND) -> None:
    grid = timestamp.reshape(2, 2)
    expected = dummy.dummy_date(timestamp).reshape(2, 2, -1)
    actual = dummy.dummy_date(grid)
    np.testing.assert_array_equal(actual, expected)
