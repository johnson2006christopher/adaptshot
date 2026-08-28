"""Semantic names for the array types this library passes around (#44).

Every numpy annotation in the package used to be a bare ``np.ndarray``, which
means ``ndarray[Any, Any]``: mypy knew a value was *an array* and nothing else.
That is not a cosmetic gap. Under numpy 2.2 -- which Python 3.10 and 3.11
resolve to on the CI matrix -- ``--strict`` reported 157 ``type-arg`` errors for
exactly this, and on one occasion the vagueness produced a *wrong* type: given an
``Any`` shape, mypy picked the ``size=None`` overload of ``rng.normal`` and
decided the result was a scalar ``float``.

**What these aliases do and do not buy.** They parameterise the *dtype*, so
passing a label array where an embedding is expected is now an error. They do
**not** constrain shape: `NDArray[T]` is `ndarray[tuple[int, ...], dtype[T]]`,
so a ``[N, D]`` matrix still type-checks where a ``[D]`` vector is expected.

Shape typing is expressible -- ``ndarray[tuple[int], ...]`` is a 1-D array -- but
not yet usable here, because numpy's own stubs do not preserve it. ``np.clip``
on a 1-D array is typed as returning an array of unknown rank, so annotating a
variable as 1-D and then clipping it is an error about numpy rather than about
us. The docstrings remain the record of shape until the stubs catch up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Embeddings, distances, similarities, probabilities -- anything measured.
#: Deliberately `np.floating[Any]` rather than a fixed width: embeddings are
#: float32 to keep the buffer small, while the calibration and conformal maths
#: accumulate in float64, and both flow through the same functions.
FloatArray = NDArray[np.floating[Any]]

#: Class labels. `object` because a label is `str | int` by the public API's
#: own signature, and numpy has no union dtype -- an array of Python objects is
#: how that is actually represented.
LabelArray = NDArray[np.object_]

#: Positions into another array: indices, counts, bin assignments.
IntArray = NDArray[np.integer[Any]]

#: Masks and predicates, kept distinct from IntArray so that `mask.sum()` and
#: `indices[mask]` cannot be confused for one another.
BoolArray = NDArray[np.bool_]

__all__ = ["BoolArray", "FloatArray", "IntArray", "LabelArray"]
