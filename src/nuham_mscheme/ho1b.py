"""Harmonic-oscillator one-body matrix elements of rho = r/b, pi = -i b grad, and T_int pieces.

Closed forms follow qchem-to-nuclear derivation 02 (Eqs. 4-6): the orbital reduced element of rho,
the (l 1/2) j recoupling through a 6j, and Wigner-Eckart in j. With b = 1, pi_mu = i s rho_mu on
matrix elements between e' = e + s (s = +-1), so T_int needs only real rho_mu blocks.
Radial phase: R_nl positive at the origin (NuHamil and imsrg++).
"""

from fractions import Fraction
from functools import cache
from math import factorial, sqrt

import numpy as np

from nuham_mscheme.cg import clebsch_gordan
from nuham_mscheme.orbits import Orbit


def _triangle(a: int, b: int, c: int) -> bool:
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def wigner_6j(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> float:
    """Return the 6j symbol {j1 j2 j3; j4 j5 j6}; all arguments doubled (Racah formula)."""
    if not (
        _triangle(j1, j2, j3)
        and _triangle(j1, j5, j6)
        and _triangle(j4, j2, j6)
        and _triangle(j4, j5, j3)
    ):
        return 0.0
    f = factorial

    def delta2(a: int, b: int, c: int) -> Fraction:
        return Fraction(
            f((a + b - c) // 2) * f((a - b + c) // 2) * f((-a + b + c) // 2),
            f((a + b + c) // 2 + 1),
        )

    pref = delta2(j1, j2, j3) * delta2(j1, j5, j6) * delta2(j4, j2, j6) * delta2(j4, j5, j3)
    t1 = ((j1 + j2 + j3) // 2, (j1 + j5 + j6) // 2, (j4 + j2 + j6) // 2, (j4 + j5 + j3) // 2)
    t2 = ((j1 + j2 + j4 + j5) // 2, (j2 + j3 + j5 + j6) // 2, (j3 + j1 + j6 + j4) // 2)
    total = Fraction(0)
    for t in range(max(t1), min(t2) + 1):
        den = 1
        for x in t1:
            den *= f(t - x)
        for x in t2:
            den *= f(x - t)
        total += Fraction((-1) ** t * f(t + 1), den)
    return float(total) * sqrt(float(pref))


def wigner_3j(j1: int, m1: int, j2: int, m2: int, j3: int, m3: int) -> float:
    """Return the 3j symbol (j1 j2 j3; m1 m2 m3), all arguments doubled."""
    if m1 + m2 + m3 != 0:
        return 0.0
    phase = (-1) ** ((j1 - j2 - m3) // 2)
    return phase / sqrt(j3 + 1) * clebsch_gordan(j1, m1, j2, m2, j3, -m3)


@cache
def rho_orbital(n1: int, l1: int, n2: int, l2: int) -> float:
    """Return <n1 l1 || rho || n2 l2>, the orbital reduced element (derivation 02 Eq. 4)."""
    e1, e2 = 2 * n1 + l1, 2 * n2 + l2
    if l1 == l2 + 1:
        if e1 == e2 + 1:
            return sqrt(l2 + 1) * sqrt(n2 + l2 + 1.5)
        if e1 == e2 - 1:
            return -sqrt(l2 + 1) * sqrt(n2)
        return 0.0
    if l1 == l2 - 1:
        return -rho_orbital(n2, l2, n1, l1)
    return 0.0


@cache
def rho_reduced_j(o1: Orbit, o2: Orbit) -> float:
    """Return <(l1 1/2) j1 || rho || (l2 1/2) j2> (derivation 02 Eq. 6)."""
    rl = rho_orbital(o1.n, o1.l, o2.n, o2.l)
    if rl == 0.0:
        return 0.0
    phase = (-1) ** ((2 * o1.l + o2.j2 + 3) // 2)
    six = wigner_6j(2 * o1.l, o1.j2, 1, o2.j2, 2 * o2.l, 2)
    return phase * sqrt((o2.j2 + 1) * (o1.j2 + 1)) * six * rl


@cache
def rho_block(o1: Orbit, o2: Orbit) -> np.ndarray:
    """Return rho_mu matrix elements <o1 m1|rho_mu|o2 m2> as [mu + 1, i1, i2], mu = m1 - m2.

    Indices i = m + j as in ``cg_table``. Real, read-only.
    """
    out = np.zeros((3, o1.j2 + 1, o2.j2 + 1))
    red = rho_reduced_j(o1, o2)
    if red != 0.0:
        for i1 in range(o1.j2 + 1):
            m1 = 2 * i1 - o1.j2
            for i2 in range(o2.j2 + 1):
                m2 = 2 * i2 - o2.j2
                if (m1 - m2) % 2 or abs(m1 - m2) > 2:
                    continue
                mu2 = m1 - m2
                out[mu2 // 2 + 1, i1, i2] = (
                    red * (-1) ** ((o1.j2 - m1) // 2) * wigner_3j(o1.j2, -m1, 2, mu2, o2.j2, m2)
                )
    out.setflags(write=False)
    return out


def kinetic_one_body(o1: Orbit, o2: Orbit, hw: float) -> float:
    """Return <o1 || p^2/2m ||o2> diagonal-in-m value (MeV): HO closed form, +sign off-diagonal."""
    if o1.l != o2.l or o1.j2 != o2.j2:
        return 0.0
    n1, n2, l = o1.n, o2.n, o1.l
    if n1 == n2:
        return 0.5 * hw * (2 * n1 + l + 1.5)
    if n1 == n2 - 1:
        return 0.5 * hw * sqrt(n2 * (n2 + l + 0.5))
    if n1 == n2 + 1:
        return 0.5 * hw * sqrt(n1 * (n1 + l + 0.5))
    return 0.0
