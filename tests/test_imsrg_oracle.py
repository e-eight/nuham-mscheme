"""Random-tuple comparison with imsrg++'s UnitTest::GetMschemeMatrixElement_2b (Phase 1 gate)."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nuham_mscheme.me2j import iter_tuples
from nuham_mscheme.mscheme2 import PN, quad_blocks
from nuham_mscheme.orbits import orbit_list

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/tmp/nuham-imsrg")

pytestmark = pytest.mark.skipif(
    shutil.which("podman") is None or not (WORK / "imsrg" / "build" / "libIMSRG.so").exists(),
    reason="needs podman and the imsrg++ build from nuham-jax refs/imsrg/build.sh",
)


def _tz(blk, which):
    if blk.sector == 0:
        return -1, -1
    if blk.sector == 2:
        return 1, 1
    assign = blk.bra_assign if which == "bra" else blk.ket_assign
    return (-1, 1) if assign == 0 else (1, -1)


def _sample(path, emax, e2max, n_quads, per_block, seed):
    rng = np.random.default_rng(seed)
    orbits = orbit_list(emax)
    tuples = list(iter_tuples(path, emax, e2max))
    picks = rng.choice(len(tuples), size=n_quads, replace=False)
    lines, expected, labels = [], [], []
    for i in picks:
        t = tuples[i]
        oa, ob, oc, od = (orbits[k] for k in (t.a, t.b, t.c, t.d))
        for blk in quad_blocks(orbits, t):
            tb, tk = _tz(blk, "bra"), _tz(blk, "ket")
            valid = np.argwhere(
                (blk.mb2[:, :, None, None] == blk.mk2[None, None])
                & blk.valid_b[:, :, None, None]
                & blk.valid_k[None, None]
            )
            if len(valid) == 0:
                continue
            for ia, ib, ic, id_ in valid[rng.integers(len(valid), size=per_block)]:
                st = []
                for o, tz, k in (
                    (oa, tb[0], ia),
                    (ob, tb[1], ib),
                    (oc, tk[0], ic),
                    (od, tk[1], id_),
                ):
                    st.append(f"{o.n} {o.l} {o.j2} {tz} {2 * k - o.j2}")
                lines.append("2  " + "  ".join(st))
                expected.append(blk.x[ia, ib, ic, id_])
                labels.append((blk.sector, t.a, t.b, t.c, t.d))
    return lines, np.array(expected), labels


@pytest.mark.parametrize("name,emax,e2max", [("e4_file", 4, 8), ("e6_file", 6, 12)])
def test_random_elements_match_imsrg(request, tmp_path, name, emax, e2max):
    path = request.getfixturevalue(name)
    lines, expected, labels = _sample(path, emax, e2max, n_quads=150, per_block=3, seed=7)
    tin, tout = tmp_path / "t.in", tmp_path / "t.out"
    tin.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [
            str(ROOT / "tools" / "run_oracle.sh"),
            str(path),
            str(emax),
            str(e2max),
            str(tin),
            str(tout),
        ],
        check=True,
        capture_output=True,
    )
    got = np.loadtxt(tout)
    assert got.shape == expected.shape
    # imsrg++ stores floats: expect agreement to ~1e-6 MeV
    err = np.abs(got - expected)
    bad = np.argsort(err)[-5:]
    assert err.max() < 2e-6, [(labels[i], expected[i], got[i]) for i in bad]
    # the sample must exercise all sectors and nonzero elements
    assert {lab[0] for lab in labels} == {0, 1, 2}
    assert (np.abs(expected) > 1e-3).sum() > 100
    assert PN == 1


def _run(path, emax, e2max, lines, tmp_path, mode, hw, a):
    import os

    tin, tout = tmp_path / "q.in", tmp_path / "q.out"
    tin.write_text("\n".join(lines) + "\n")
    env = {**os.environ, "MODE": mode, "HW": str(hw), "A": str(a)}
    subprocess.run(
        [
            str(ROOT / "tools" / "run_oracle.sh"),
            str(path),
            str(emax),
            str(e2max),
            str(tin),
            str(tout),
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    return np.loadtxt(tout)


@pytest.mark.parametrize("hw,a", [(16.0, 16), (20.0, 4)])
def test_total_two_body_with_tint_matches_imsrg(e6_file, tmp_path, hw, a):
    """V (file) + two-body T_int against imsrg++ file + Trel_Op, element by element."""
    from nuham_mscheme.mscheme2 import TInt

    rng = np.random.default_rng(3)
    orbits = orbit_list(6)
    tuples = list(iter_tuples(e6_file, 6, 12))
    lines, expected = [], []
    for i in rng.choice(len(tuples), size=400, replace=False):
        t = tuples[i]
        for blk in quad_blocks(orbits, t, tint=TInt(hw, a)):
            tb, tk = _tz(blk, "bra"), _tz(blk, "ket")
            valid = np.argwhere(
                (blk.mb2[:, :, None, None] == blk.mk2[None, None])
                & blk.valid_b[:, :, None, None]
                & blk.valid_k[None, None]
            )
            if len(valid) == 0:
                continue
            for ia, ib, ic, id_ in valid[rng.integers(len(valid), size=3)]:
                os_ = (orbits[t.a], orbits[t.b], orbits[t.c], orbits[t.d])
                tzs = (tb[0], tb[1], tk[0], tk[1])
                ks = (ia, ib, ic, id_)
                lines.append(
                    "2  "
                    + "  ".join(
                        f"{o.n} {o.l} {o.j2} {tz} {2 * k - o.j2}"
                        for o, tz, k in zip(os_, tzs, ks, strict=True)
                    )
                )
                expected.append(blk.x[ia, ib, ic, id_])
    expected = np.array(expected)
    got = _run(e6_file, 6, 12, lines, tmp_path, "file+trel", hw, a)
    err = np.abs(got - expected)
    assert err.max() < 5e-6, err.max()
    # the T_int part must matter: compare with the file-only value of the same elements
    only_file = _run(e6_file, 6, 12, lines, tmp_path, "file", hw, a)
    assert np.abs(got - only_file).max() > 0.05


def test_one_body_tint_matches_imsrg(e6_file, tmp_path):
    from nuham_mscheme.onebody import tint_one_body

    hw, a = 16.0, 16
    orbits = orbit_list(6)
    lines, expected = [], []
    for o1 in orbits:
        for o2 in orbits:
            # m = +j exists for both orbits when j agrees; other pairs are zero by j conservation
            if o1.j2 != o2.j2:
                continue
            for tz in (-1, 1):
                st = [f"{o.n} {o.l} {o.j2} {tz} {o.j2}" for o in (o1, o2)]
                lines.append("1  " + "  ".join(st))
                expected.append(tint_one_body(o1, o2, hw, a))
    got = _run(e6_file, 6, 12, lines, tmp_path, "trel", hw, a)
    assert np.abs(got - np.array(expected)).max() < 1e-5
    assert (np.abs(expected) > 1.0).sum() > 20
