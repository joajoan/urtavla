"""Date features for NumPy `datetime64` timestamps.

The package turns timestamps into numbers that a model can use directly.
The parts of a date that repeat become positions on a circle, and a few
notable days become flags.

The name "urtavla" is Swedish for a clock face. A clock face shows the
time on several dials, and each feature of this package can be viewed as
one such dial.
"""

from urtavla.cycle import cyclic_date, cyclic_year, dyadic_period
from urtavla.dummy import dummy_date

__all__ = ("cyclic_date", "cyclic_year", "dummy_date", "dyadic_period")
