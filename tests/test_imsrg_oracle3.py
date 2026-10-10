"""Random-element comparison with imsrg++'s UnitTest::GetMschemeMatrixElement_3b (Phase 2 gate)."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nuham_mscheme.me3j import Layout, filter_me3j, iter_tuples3
from nuham_mscheme.mscheme3 import TripleBasis, triple_blocks

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/tmp/nuham-imsrg")
REAL = ROOT / "runs/inputs/3n/N3LO_N3LOEMlnl_hw30to16_e8_e2_16_e3_8.me3j.gz"

pytestmark = pytest.mark.skipif(
    shutil.which("podman") is None or not (WORK / "imsrg" / "build" / "libIMSRG.so").exists(),
    reason="needs podman and the imsrg++ build from nuham-jax refs/imsrg/build.sh",
)


def _kind(t: tuple[int, int, int]) -> str:
    """Coincidence pattern of an orbit triple: distinct, a==b, b==c or a==b==c."""
    if t[0] == t[1] == t[2]:
        return "abc"
    return "ab" if t[0] == t[1] else "bc" if t[1] == t[2] else "distinct"


def _state(basis: TripleBasis, g: int) -> str:
    species, orb, m2 = basis.decode(g)
    o = basis.orbits[orb]
    return f"{o.n} {o.l} {o.j2} {2 * species - 1} {m2}"


def _sample(basis, tuples, per_bucket, per_tuple, seed):
    """Stratified random elements: lines for the oracle, our values, and a label per element."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, str], list[int]] = {}
    for i, t in enumerate(tuples):
        buckets.setdefault((_kind(t.bra), _kind(t.ket)), []).append(i)
    lines, expected, labels = [], [], []
    for key, idx in sorted(buckets.items()):
        for i in rng.choice(idx, size=min(per_bucket, len(idx)), replace=False):
            blocks = list(triple_blocks(basis, tuples[i]))
            for b in rng.choice(len(blocks), size=min(per_tuple, len(blocks)), replace=False):
                blk = blocks[b]
                for _ in range(3):
                    r = rng.integers(blk.x.shape[0])
                    c = rng.integers(blk.x.shape[1])
                    bra = " ".join(_state(basis, g) for g in blk.bra_g[r])
                    ket = " ".join(_state(basis, g) for g in blk.ket_g[c])
                    lines.append(f"{bra}  {ket}")
                    expected.append(blk.x[r, c])
                    labels.append((*key, blk.sector, blk.m2))
    return lines, np.array(expected), labels


@pytest.mark.parametrize("limits", [(4, 8, 4), (6, 12, 6), (8, 16, 8)])
def test_random_elements_match_imsrg(tmp_path, limits):
    if not REAL.exists():
        pytest.skip(f"production file missing: {REAL}")
    path = tmp_path / "cut.me3j"
    filter_me3j(REAL, path, (16, 8), limits)
    tuples = list(iter_tuples3(path, *limits))
    basis = TripleBasis(Layout(*limits).orbits)
    lines, expected, labels = _sample(basis, tuples, per_bucket=5, per_tuple=4, seed=11)
    tin, tout = tmp_path / "q.in", tmp_path / "q.out"
    tin.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [str(ROOT / "tools" / "run_oracle3.sh"), str(path), *map(str, limits), str(tin), str(tout)],
        check=True,
        capture_output=True,
    )
    got = np.loadtxt(tout)
    assert got.shape == expected.shape
    err = np.abs(got - expected)
    bad = np.argsort(err)[-5:]
    # imsrg++ keeps the file's values as floats; measured agreement is a few 1e-9 MeV
    assert err.max() < 5e-8, [(labels[i], expected[i], got[i]) for i in bad]
    big = np.abs(expected) > 1e-2
    assert big.sum() > 100
    assert np.allclose(got[big] / expected[big], 1.0, atol=1e-6)  # no stray normalisation factor
    assert {lab[2] for lab in labels} == {0, 1, 2, 3}
    assert {lab[:2] for lab in labels} >= {("abc", "abc"), ("ab", "bc"), ("distinct", "distinct")}
