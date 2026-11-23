from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from urtavla import cycle, encode

if TYPE_CHECKING:
    from urtavla.typing import DtND, FloatND


@pytest.fixture
def flag() -> FloatND:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )


def test_encode_date(timestamp: DtND, flag: FloatND) -> None:
    period = cycle.dyadic_period(1, 2)
    cyclic = cycle.cyclic_date(timestamp, period)
    expected = np.concatenate([cyclic, flag], axis=-1)
    actual = encode.encode_date(timestamp, period)
    np.testing.assert_array_equal(actual, expected)


def test_encode_date_keeps_leading_shape(timestamp: DtND) -> None:
    grid = timestamp.reshape(2, 2)
    period = cycle.dyadic_period(1, 2)
    expected = encode.encode_date(timestamp, period).reshape(2, 2, -1)
    actual = encode.encode_date(grid, period)
    np.testing.assert_array_equal(actual, expected)
