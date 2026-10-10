"""Exact Clebsch-Gordan tables for the m-scheme expansion."""

from fractions import Fraction
from functools import cache
from math import factorial, sqrt

import numpy as np


def clebsch_gordan(j1x2: int, m1x2: int, j2x2: int, m2x2: int, jx2: int, mx2: int) -> float:
    """Return <j1 m1 j2 m2 | J M> (Condon-Shortley), all arguments doubled.

    The Racah sum is evaluated in exact rational arithmetic and converted at the end.

    Parameters
    ----------
    j1x2, m1x2, j2x2, m2x2, jx2, mx2 : int
        Twice the angular momenta and projections.

    Returns
    -------
    float
        The coefficient; zero when a selection rule fails.
    """
    if m1x2 + m2x2 != mx2 or abs(m1x2) > j1x2 or abs(m2x2) > j2x2 or abs(mx2) > jx2:
        return 0.0
    if not abs(j1x2 - j2x2) <= jx2 <= j1x2 + j2x2 or (j1x2 + j2x2 + jx2) % 2:
        return 0.0
    f = factorial
    a = (j1x2 + j2x2 - jx2) // 2
    b = (jx2 + j1x2 - j2x2) // 2
    c = (jx2 - j1x2 + j2x2) // 2
    d = (j1x2 + j2x2 + jx2) // 2 + 1
    pref = Fraction((jx2 + 1) * f(a) * f(b) * f(c), f(d))
    pref *= Fraction(
        f((jx2 + mx2) // 2)
        * f((jx2 - mx2) // 2)
        * f((j1x2 - m1x2) // 2)
        * f((j1x2 + m1x2) // 2)
        * f((j2x2 - m2x2) // 2)
        * f((j2x2 + m2x2) // 2)
    )
    total = Fraction(0)
    for k in range(0, a + 1):
        args = (
            k,
            a - k,
            (j1x2 - m1x2) // 2 - k,
            (j2x2 + m2x2) // 2 - k,
            (jx2 - j2x2 + m1x2) // 2 + k,
            (jx2 - j1x2 - m2x2) // 2 + k,
        )
        if min(args) < 0:
            continue
        den = 1
        for x in args:
            den *= f(x)
        total += Fraction((-1) ** k, den)
    return float(total) * sqrt(float(pref))


@cache
def cg_table(j1x2: int, j2x2: int, jx2: int) -> np.ndarray:
    """Return <j1 m1 j2 m2 | J M> as an array [m1 index, m2 index], M = m1 + m2.

    The m index runs over m = -j..j in steps of 1: position i has m = i - j, i.e. 2m = 2i - 2j.

    Parameters
    ----------
    j1x2, j2x2, jx2 : int
        Twice the angular momenta.

    Returns
    -------
    numpy.ndarray
        Shape (j1x2 + 1, j2x2 + 1), read-only.
    """
    t = np.zeros((j1x2 + 1, j2x2 + 1))
    for i1 in range(j1x2 + 1):
        for i2 in range(j2x2 + 1):
            m1x2, m2x2 = 2 * i1 - j1x2, 2 * i2 - j2x2
            t[i1, i2] = clebsch_gordan(j1x2, m1x2, j2x2, m2x2, jx2, m1x2 + m2x2)
    t.setflags(write=False)
    return t
