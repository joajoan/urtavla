# urtavla

This package, `urtavla`, turns NumPy `datetime64` timestamps into numeric
features that a model can use directly.

A raw date is awkward for a model: 31 December and 1 January are one day apart,
yet their day numbers are 365 and 1. urtavla instead places each date on a
circle, as a sine and a cosine, so that the end of every week, month and year
lies next to its start. It also provides a few yes-or-no flags, such as whether
the day falls on a weekend.

The name is Swedish for a clock face.

## Installation

```sh
uv add git+https://github.com/joajoan/urtavla
```

The package needs Python 3.12 or later, and NumPy is its only dependency.

## Usage

Every function below takes timestamps of any shape and resolution. The features
are returned along a new last axis, so a result has the shape of its input plus
one axis. The examples all use the same two dates.

```python
>>> import numpy as np
>>> import urtavla
>>> ts = np.array(["2024-02-29", "2024-12-30"], dtype="datetime64[D]")

```

### `cyclic_date`

The `cyclic_date` function returns the position of each date within six cycles:
the day of the year, the weekday, the day of the month, the month, and the week
and the day of the ISO 8601 year. Each position is given as a sine and a cosine,
which makes twelve columns. A position is divided by the true length of its
cycle, so February and March both run the full circle.

```python
>>> cyclic = urtavla.cyclic_date(ts)
>>> cyclic.shape
(2, 12)

```

### `dummy_date`

The `dummy_date` function returns four flags for each date. The flags mark 29
February, the weekend, the first day of a month and the last day of a month, in
that order. In the example, 29 February 2024 is both a leap day and the last day
of its month, while 30 December 2024 is none of the four.

```python
>>> dummy = urtavla.dummy_date(ts)
>>> dummy
array([[ True, False, False,  True],
       [False, False, False, False]])

```

### `cyclic_year`

The year number does not repeat, so it has no cycle of its own. The
`cyclic_year` function therefore measures the year against periods that you
choose, and returns a sine and a cosine for each period. Long periods keep
nearby years close together, while short periods tell them apart.

The `dyadic_period` function builds such periods. It takes the exponents of the
shortest and the longest period, and returns every power of two between them, in
years.

```python
>>> period = urtavla.dyadic_period(1, 5)
>>> period
array([ 2.,  4.,  8., 16., 32.])
>>> year = urtavla.cyclic_year(ts, period)
>>> year.shape
(2, 10)

```

### `CyclicYear`

The `CyclicYear` class holds the periods for `cyclic_year`. It suits the case
where the periods are chosen once and reused, because an instance is called with
the timestamps alone.

```python
>>> encoder = urtavla.CyclicYear.from_dyadic(1, 5)
>>> encoder.period
array([ 2.,  4.,  8., 16., 32.])
>>> encoder(ts).shape
(2, 10)

```

### Feature matrix

All results have the shape of the timestamps in every axis but the last, so they
join along the last axis into one matrix.

```python
>>> feature = np.concatenate([cyclic, year, dummy], axis=-1)
>>> feature.shape
(2, 26)

```

## Development

The project is managed with [uv](https://docs.astral.sh/uv/), and its recipes
run through [just](https://just.systems/):

```sh
just check   # lint with ruff, type-check with ty, and check the Markdown
just test    # run pytest, including the examples in the docstrings and here
just format  # fix lint errors, format the code, and wrap the Markdown
```
