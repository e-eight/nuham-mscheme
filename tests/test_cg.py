import itertools

import pytest
from sympy import Rational, S
from sympy.physics.quantum.cg import CG

from nuham_mscheme.cg import clebsch_gordan


@pytest.mark.parametrize("j1x2,j2x2", [(1, 1), (1, 3), (3, 5), (5, 9), (17, 17), (7, 15)])
def test_cg_matches_sympy(j1x2, j2x2):
    for jx2 in range(abs(j1x2 - j2x2), j1x2 + j2x2 + 1, 2):
        for m1x2 in range(-j1x2, j1x2 + 1, 2):
            for m2x2 in range(-j2x2, j2x2 + 1, 2):
                mx2 = m1x2 + m2x2
                if abs(mx2) > jx2:
                    continue
                ref = CG(
                    Rational(j1x2, 2),
                    Rational(m1x2, 2),
                    Rational(j2x2, 2),
                    Rational(m2x2, 2),
                    Rational(jx2, 2),
                    Rational(mx2, 2),
                ).doit()
                assert clebsch_gordan(j1x2, m1x2, j2x2, m2x2, jx2, mx2) == pytest.approx(
                    float(ref), abs=1e-14
                )


def test_cg_selection_rules():
    assert clebsch_gordan(1, 1, 1, 1, 4, 2) == 0.0
    assert clebsch_gordan(1, 1, 1, -1, 2, 0) != 0.0
    assert S(0) == 0 and itertools is not None
