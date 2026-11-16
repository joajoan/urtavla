from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from urtavla import cycle

if TYPE_CHECKING:
    from urtavla.typing import DtND, FloatND


def test_unit_circle() -> None:
    # A twelfth of a turn: at a quarter turn the cosine is zero, so a
    # flipped sign would pass.
    turn = np.array(1 / 12)
    actual = cycle.unit_circle(turn)
    np.testing.assert_allclose(actual, [0.5, 3**0.5 / 2], atol=1e-9)


@pytest.fixture
def calendar_day_of_year() -> FloatND:
    turn = np.array([365 / 366, 59 / 366, 0 / 365, 2 / 365])
    return cycle.unit_circle(turn)


def test_cyclic_calendar_day_of_year(
    timestamp: DtND, calendar_day_of_year: FloatND
) -> None:
    actual = cycle.cyclic_calendar_day_of_year(timestamp)
    np.testing.assert_allclose(actual, calendar_day_of_year, atol=1e-9)


@pytest.fixture
def day_of_week() -> FloatND:
    turn = np.array([1 / 7, 5 / 7, 4 / 7, 6 / 7])
    return cycle.unit_circle(turn)


def test_cyclic_day_of_week(timestamp: DtND, day_of_week: FloatND) -> None:
    actual = cycle.cyclic_day_of_week(timestamp)
    np.testing.assert_allclose(actual, day_of_week, atol=1e-9)


@pytest.fixture
def calendar_day_of_month() -> FloatND:
    turn = np.array([30 / 31, 28 / 29, 0 / 31, 2 / 31])
    return cycle.unit_circle(turn)


def test_cyclic_calendar_day_of_month(
    timestamp: DtND, calendar_day_of_month: FloatND
) -> None:
    actual = cycle.cyclic_calendar_day_of_month(timestamp)
    np.testing.assert_allclose(actual, calendar_day_of_month, atol=1e-9)


@pytest.fixture
def calendar_month() -> FloatND:
    turn = np.array([11 / 12, 1 / 12, 0 / 12, 0 / 12])
    return cycle.unit_circle(turn)


def test_cyclic_calendar_month(
    timestamp: DtND, calendar_month: FloatND
) -> None:
    actual = cycle.cyclic_calendar_month(timestamp)
    np.testing.assert_allclose(actual, calendar_month, atol=1e-9)


@pytest.fixture
def iso_week_of_year() -> FloatND:
    turn = np.array([0 / 52, 8 / 53, 52 / 53, 52 / 53])
    return cycle.unit_circle(turn)


def test_cyclic_iso_week_of_year(
    timestamp: DtND, iso_week_of_year: FloatND
) -> None:
    actual = cycle.cyclic_iso_week_of_year(timestamp)
    np.testing.assert_allclose(actual, iso_week_of_year, atol=1e-9)


@pytest.fixture
def iso_day_of_year() -> FloatND:
    turn = np.array([1 / 364, 61 / 371, 368 / 371, 370 / 371])
    return cycle.unit_circle(turn)


def test_cyclic_iso_day_of_year(
    timestamp: DtND, iso_day_of_year: FloatND
) -> None:
    actual = cycle.cyclic_iso_day_of_year(timestamp)
    np.testing.assert_allclose(actual, iso_day_of_year, atol=1e-9)


@pytest.fixture
def cyclic_array(
    calendar_day_of_year: FloatND,
    day_of_week: FloatND,
    calendar_day_of_month: FloatND,
    calendar_month: FloatND,
    iso_week_of_year: FloatND,
    iso_day_of_year: FloatND,
) -> FloatND:
    return np.concatenate(
        [
            calendar_day_of_year,
            day_of_week,
            calendar_day_of_month,
            calendar_month,
            iso_week_of_year,
            iso_day_of_year,
        ],
        axis=-1,
    )


def test_cyclic_date(timestamp: DtND, cyclic_array: FloatND) -> None:
    actual = cycle.cyclic_date(timestamp)
    np.testing.assert_allclose(actual, cyclic_array, atol=1e-9)


def test_cyclic_date_keeps_leading_shape(timestamp: DtND) -> None:
    grid = timestamp.reshape(2, 2)
    expected = cycle.cyclic_date(timestamp).reshape(2, 2, -1)
    actual = cycle.cyclic_date(grid)
    np.testing.assert_array_equal(actual, expected)


def test_cyclic_year(timestamp: DtND) -> None:
    expected = [
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
    ]
    period = cycle.dyadic_period(1, 2)
    actual = cycle.cyclic_year(timestamp, period)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_cyclic_year_keeps_leading_shape(timestamp: DtND) -> None:
    grid = timestamp.reshape(2, 2)
    period = cycle.dyadic_period(1, 2)
    expected = cycle.cyclic_year(timestamp, period).reshape(2, 2, -1)
    actual = cycle.cyclic_year(grid, period)
    np.testing.assert_array_equal(actual, expected)
