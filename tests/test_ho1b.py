import itertools

import pytest
from sympy import Rational
from sympy.physics.wigner import wigner_3j as s3j
from sympy.physics.wigner import wigner_6j as s6j

from nuham_mscheme.ho1b import rho_block, wigner_3j, wigner_6j
from nuham_mscheme.orbits import orbit_list


def test_6j_matches_sympy():
    for j in itertools.product(range(0, 6), repeat=6):
        ref = float(s6j(*[Rational(x, 2) for x in j])) if _ok(j) else 0.0
        assert wigner_6j(*j) == pytest.approx(ref, abs=1e-13)


def _ok(j):
    j1, j2, j3, j4, j5, j6 = j
    return all(
        abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0
        for a, b, c in ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3))
    )


def test_3j_matches_sympy():
    for j1, j2, j3 in [(1, 2, 3), (3, 2, 5), (5, 2, 5), (7, 2, 9)]:
        for m1 in range(-j1, j1 + 1, 2):
            for m2 in (-2, 0, 2):
                m3 = -m1 - m2
                if abs(m3) > j3:
                    continue
                ref = float(s3j(*[Rational(x, 2) for x in (j1, j2, j3, m1, m2, m3)]))
                assert wigner_3j(j1, m1, j2, m2, j3, m3) == pytest.approx(ref, abs=1e-13)


def test_rho_is_hermitian_vector():
    orbs = orbit_list(4)
    for a, oa in enumerate(orbs):
        for c, oc in enumerate(orbs):
            blk, back = rho_block(oa, oc), rho_block(oc, oa)
            for mu in (-1, 0, 1):
                # rho_mu^dagger = (-1)^mu rho_{-mu}
                lhs = blk[mu + 1]
                rhs = (-1) ** mu * back[-mu + 1].T
                assert abs(lhs - rhs).max() < 1e-12, (a, c, mu)
