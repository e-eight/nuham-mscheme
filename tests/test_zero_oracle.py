"""Independent checks of exact frozen-coefficient evaluation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sympy import Rational, sqrt

from nuham_mscheme.me3j import Layout
from nuham_mscheme.mscheme3 import TripleBasis


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "tools" / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stored_contraction_cancellation():
    oracle = module("analyze_zero3")
    # The sequential float sum loses the unit; exact rational arithmetic must retain it.
    assert (
        oracle.exact_stored_contraction(np.ones(3), np.diag([2.0**54, 1.0, -(2.0**54)]), np.ones(3))
        == 1
    )


def test_exact_couplings_against_existing_explicit_and_invariants():
    oracle, d = module("analyze_zero3"), module("diagnose_zeros3")
    data = d.load(Path(__file__).resolve().parents[1] / "tools/fixtures/zero3-laptop.npz")
    tuples = d.unpack(data)
    basis = TripleBasis(Layout(2, 4, 4).orbits)
    table = {(t.bra, t.ket): t for t in tuples}
    for bra, ket in (([5, 7, 28], [8, 9, 21]), ([0, 8, 9], [4, 6, 9]), ([0, 1, 20], [0, 1, 20])):
        exact = oracle.exact_element(basis, tuples, bra, ket)
        old = d.original._explicit(basis, table, tuple(bra), tuple(ket))
        assert float(exact) == pytest.approx(old, abs=1e-12, rel=1e-12)
        assert oracle.exact_element(basis, tuples, ket, bra) == exact
    assert oracle.exact_element(basis, tuples, [4, 7, 28], [8, 9, 21]) == 0  # different M
    # A nonzero off-diagonal element supplies a sign-sensitive control.
    checked = False
    for block in d.kernel.triple_blocks(basis, tuples[79]):
        candidates = np.argwhere(block.x < -0.1)
        if len(candidates):
            r, c = candidates[0]
            bra, ket = block.bra_g[r].tolist(), block.ket_g[c].tolist()
            exact = oracle.exact_element(basis, tuples, bra, ket)
            assert float(exact) == pytest.approx(block.x[r, c], rel=1e-12, abs=1e-12)
            assert float(-exact) != pytest.approx(block.x[r, c], rel=1e-12, abs=1e-12)
            checked = True
            break
    assert checked
    # Standalone CG convention check guards an argument-order/sign swap.
    assert (
        oracle.clebsch_gordan(Rational(1, 2), Rational(1, 2), 0, Rational(1, 2), -Rational(1, 2), 0)
        == sqrt(2) / 2
    )


def test_frozen_arm_cancellation_and_counting():
    """Exact CG proves nonzero; frozen counterexample exercises repaired counting.

    No requirement that scalar/batched zero masks differ on every machine.
    The exact result uses stored coefficients, not an ideal unrounded operator.
    """
    oracle, d = module("analyze_zero3"), module("diagnose_zeros3")
    data = d.load(Path(__file__).resolve().parents[1] / "tools/fixtures/zero3-arm.npz")
    tuples = d.unpack(data)
    basis = TripleBasis(Layout(2, 4, 4).orbits)
    for bra, ket in (
        ([5, 7, 28], [8, 9, 21]),
        ([1, 28, 29], [8, 25, 27]),
        ([0, 8, 9], [4, 6, 9]),
        ([20, 28, 29], [24, 26, 29]),
    ):
        exact = oracle.exact_element(basis, tuples, bra, ket)
        assert exact.is_zero is False
        assert oracle.exact_element(basis, tuples, ket, bra) == exact
    blocks = d.dense_blocks(basis, tuples)
    d.original.test_streaming_stats_match_dense((basis, tuples, blocks, 1e-9))


@pytest.mark.parametrize("fault", ["counts", "elements"])
def test_repaired_checks_reject_corruption(fault, monkeypatch):
    """Mutation checks: a count error and a sign error each fail the repaired gate."""
    d = module("diagnose_zeros3")
    data = d.load(Path(__file__).resolve().parents[1] / "tools/fixtures/zero3-arm.npz")
    tuples = d.unpack(data)
    basis = TripleBasis(Layout(2, 4, 4).orbits)
    blocks = d.dense_blocks(basis, tuples)
    if fault == "counts":
        original = d.original.ThreeBodyStats.blocks

        def corrupt(self):
            result = original(self)
            next(iter(result.values())).nnz_full[0] += 1
            return result

        monkeypatch.setattr(d.original.ThreeBodyStats, "blocks", corrupt)
    else:
        # A symmetric sign flip preserves abs norms and counts, so only the
        # independent element comparison detects it.
        mat = next(mat for mat, _ in blocks.values() if (np.abs(mat) > 0.1).any())
        r, c = np.argwhere(np.abs(mat) > 0.1)[0]
        mat[r, c] = -mat[r, c]
        mat[c, r] = mat[r, c]
    with pytest.raises(AssertionError):
        d.original.test_streaming_stats_match_dense((basis, tuples, blocks, 1e-9))
