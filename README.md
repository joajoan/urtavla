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

### `cyclic_date`

The `cyclic_date` function returns the whole date as cycles. It first gives the
position of each date within six cycles: the day of the year, the weekday, the
day of the month, the month, and the week and the day of the ISO 8601 year. Each
position is given as a sine and a cosine, which makes twelve columns. A position
is divided by the true length of its cycle, so February and March both run the
full circle. The columns of `cyclic_year` follow, for the periods that you pass.

```python
>>> cyclic = urtavla.cyclic_date(ts, period)
>>> cyclic.shape
(2, 22)

```

### `CyclicDate`

The `CyclicDate` class holds the periods for `cyclic_date`, in the same way that
`CyclicYear` holds them for `cyclic_year`.

```python
>>> encoder = urtavla.CyclicDate.from_dyadic(1, 5)
>>> np.array_equal(encoder(ts), cyclic)
True

```

### `dummy_date`

The `dummy_date` function returns five flags for each date. The flags mark a
leap year, 29 February, the weekend, the first day of a month and the last day
of a month, in that order. In the example, both dates fall in 2024, which is a
leap year. The first date is also a leap day and the last day of its month,
while the second is none of the other four.

```python
>>> dummy = urtavla.dummy_date(ts)
>>> dummy
array([[ True,  True, False, False,  True],
       [ True, False, False, False, False]])

```

### `encode_date`

The `encode_date` function returns every feature above as one matrix. The cycles
of `cyclic_date` come first, and the flags of `dummy_date` follow as zeros and
ones, because a matrix has a single type. The result equals the two joined along
the last axis.

```python
>>> feature = urtavla.encode_date(ts, period)
>>> feature.shape
(2, 27)
>>> np.array_equal(feature, np.concatenate([cyclic, dummy], axis=-1))
True

```

### `EncodeDate`

The `EncodeDate` class holds the periods for `encode_date`, in the same way that
`CyclicDate` holds them for `cyclic_date`.

```python
>>> encoder = urtavla.EncodeDate.from_dyadic(1, 5)
>>> np.array_equal(encoder(ts), feature)
True

```

## Development

The project is managed with [uv](https://docs.astral.sh/uv/), and its recipes
run through [just](https://just.systems/):

```sh
just check   # lint with ruff, type-check with ty, and check the Markdown
just test    # run pytest, including the examples in the docstrings and here
just format  # fix lint errors, format the code, and wrap the Markdown
```
