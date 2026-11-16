from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from urtavla.typing import DtND


@pytest.fixture
def timestamp() -> DtND:
    """Four dates that between them reach every calendar edge.

    - Tuesday before the epoch, at minute resolution: negative counts
      round down, the last day of a leap year, and a late December that
      belongs to the next ISO year, one of 52 weeks.
    - Saturday leap day, within an ISO year of 53 weeks.
    - Friday that starts a month and a year, yet belongs to the last
      week of the previous ISO year.
    - Sunday closing that same week, which only a week starting on
      Monday keeps there, on the last day of the ISO year.
    """
    return np.array(
        [
            "1968-12-31T23:59",
            "2020-02-29T00:00",
            "2021-01-01T00:00",
            "2021-01-03T00:00",
        ],
        dtype="datetime64[m]",
    )
