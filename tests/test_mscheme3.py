"""Tests of the three-body m-scheme expansion: structure, invariance, statistics, explicit sums."""

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sympy import Rational
from sympy.physics.quantum.cg import CG

from nuham_mscheme.dense3 import dense_blocks
from nuham_mscheme.me3j import (
    Layout,
    Me3jTuple,
    filter_me3j,
    iter_tuples3,
    kept_intervals,
    tuple_entries,
)
from nuham_mscheme.mscheme3 import (
    SECTORS3,
    TripleBasis,
    _isospin,
    _slots,
    _spatial,
    triple_blocks,
    tuple_tensor,
)
from nuham_mscheme.stats3 import ThreeBodyStats

REAL = (
    Path(__file__).resolve().parents[1]
    / "runs/inputs/3n/N3LO_N3LOEMlnl_hw30to16_e8_e2_16_e3_8.me3j.gz"
)
LIMITS = (2, 4, 4)  # emax, e2max, e3max of the test space


def _antisymmetrizer(tri, j123, j12s, js) -> np.ndarray:
    """Return Pi[y, iT, x, t, X, s]: projector on the states antisymmetric under the slot
    permutations that exchange equal orbits, in the coupled (j12, t12) labels at fixed (J, T)."""
    perms = [
        p for p in itertools.permutations(range(3)) if all(tri[p[i]] == tri[i] for i in range(3))
    ]
    sp = _spatial(j123, j12s, js)
    iso, _ = _isospin()
    pi = np.zeros((len(js), 2, len(j12s), 2, len(j12s), 2))
    for p in perms:
        sgn = (-1) ** sum(p[i] > p[k] for i, k in itertools.combinations(range(3), 2))
        pinv = [p.index(k) for k in range(3)]
        spp = np.transpose(sp, (*p, 3, 4))
        psp = np.einsum("abcxy,abcXy->yxX", sp, spp) / (np.array(js) + 1)[:, None, None]
        bits = [[(a >> 2) & 1, (a >> 1) & 1, a & 1] for a in range(8)]
        perm_assign = [sum(bits[a][pinv[k]] << (2 - k) for k in range(3)) for a in range(8)]
        piso = (
            np.einsum("atT,asT->Tts", iso, iso[perm_assign]) / np.array([2.0, 4.0])[:, None, None]
        )
        pi += sgn / len(perms) * np.einsum("yxX,Tts->yTxtXs", psp, piso)
    return pi


def _random_values(lay: Layout, bra: int, ket: int, rng: np.random.Generator) -> np.ndarray:
    """Random file values of a physical operator: antisymmetric, Hermitian (bra == ket)."""
    jb, jk = lay.jkeys[bra], lay.jkeys[ket]
    j12s, j45s, js, index = _slots(jb, jk)
    vt = np.zeros((len(j12s), 2, len(j45s), 2, len(js), 2))
    vt[tuple(index.T)] = rng.normal(size=len(index))
    if bra == ket:
        vt = 0.5 * (vt + vt.transpose(2, 3, 0, 1, 4, 5))
    pib = _antisymmetrizer(lay.triples[bra], jb, j12s, js)
    pik = _antisymmetrizer(lay.triples[ket], jk, j45s, js)
    step = np.einsum("yTxtXs,XskuyT->xtkuyT", pib, vt)
    vt = np.einsum("xtKUyT,yTKUku->xtkuyT", step, pik)
    v = vt[tuple(index.T)]
    v[np.abs(v) < 1e-12] = 0.0
    return v


def _synthetic() -> list[Me3jTuple]:
    lay = Layout(LIMITS[0], LIMITS[1], LIMITS[2])
    rng = np.random.default_rng(20261010)
    keep = np.ones(len(lay.triples), dtype=bool)
    return [
        Me3jTuple(lay.triples[i], lay.triples[k], _random_values(lay, i, k, rng))
        for _, _, i, k in kept_intervals(lay, keep)
    ]


def _real(tmp_path_factory: pytest.TempPathFactory) -> list[Me3jTuple]:
    if not REAL.exists():
        pytest.skip(f"production file missing: {REAL}")
    dst = tmp_path_factory.mktemp("real3") / "small.me3j"
    filter_me3j(REAL, dst, (16, 8), LIMITS)
    return list(iter_tuples3(dst, *LIMITS))


@pytest.fixture(scope="module", params=["synthetic", "real"])
def space(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    tuples = _synthetic() if request.param == "synthetic" else _real(tmp_path_factory)
    basis = TripleBasis(Layout(LIMITS[0], LIMITS[1], LIMITS[2]).orbits)
    # real files carry 8 decimals, so invariances hold only to about 1e-8 MeV
    tol = 1e-9 if request.param == "synthetic" else 1e-7
    return basis, tuples, dense_blocks(basis, tuples), tol


def _sp_states(basis: TripleBasis) -> list[tuple[int, int, int]]:
    return [basis.decode(g) for g in range(2 * basis.nsp)]


def test_global_order_is_protons_first_then_orbit_then_m(space):
    basis, *_ = space
    sps = _sp_states(basis)
    assert sps == sorted(sps, key=lambda s: (s[0], s[1], s[2]))
    assert all(sps[g] == basis.decode(g) for g in range(len(sps)))


def test_every_triple_state_appears_once(space):
    """Rows of all species assignments of an orbit triple are exactly its unique p<q<r states."""
    basis, tuples, *_ = space
    sps = _sp_states(basis)
    expected: Counter = Counter()
    for tri in itertools.combinations(range(len(sps)), 3):
        expected[tuple(sorted((sps[g][1] for g in tri), reverse=True))] += 1
    seen = set()
    for t in tuples:
        if t.bra != t.ket or t.bra in seen:
            continue
        seen.add(t.bra)
        rows = []
        for assign in range(8):
            st = basis.slot_states(t.bra, assign)
            rows += [tuple(g) for g in st.g[st.ok].tolist()]
        assert len(rows) == len(set(rows)) == expected[t.bra], t.bra
        assert all(g[0] < g[1] < g[2] for g in rows)


def test_hermitian_and_written_once(space):
    """Diagonal-tuple blocks are symmetric as generated; dense_blocks raises on a double write."""
    _, _, blocks, _ = space
    big = max(np.abs(m).max() for m, _ in blocks.values())
    assert big > 0.1
    for key, (mat, _) in blocks.items():
        assert np.abs(mat - mat.T).max() < 1e-12 * max(big, 1.0), key


def _lowering_j(basis: TripleBasis, blocks, sector: int, m2: int) -> np.ndarray:
    """J_- from block (sector, m2) to (sector, m2 - 2) on triple states."""
    src, dst = blocks[(sector, m2)][1], blocks[(sector, m2 - 2)][1]
    pos = {g: i for i, g in enumerate(dst)}
    lm = np.zeros((len(dst), len(src)))
    for col, tri in enumerate(src):
        for slot in range(3):
            sp, orb, m = basis.decode(tri[slot])
            j2 = basis.orbits[orb].j2
            if m - 2 < -j2:
                continue
            new = list(tri)
            new[slot] -= 1
            if len(set(new)) < 3:
                continue
            inv = sum(new[i] > new[k] for i, k in itertools.combinations(range(3), 2))
            lm[pos[tuple(sorted(new))], col] += (-1) ** inv * np.sqrt(
                (j2 + m) / 2 * (j2 - m + 2) / 2
            )
    return lm


def test_rotational_invariance(space):
    basis, _, blocks, tol = space
    checked = 0
    for (s, m2), (mat, _) in blocks.items():
        if (s, m2 - 2) not in blocks:
            continue
        lm = _lowering_j(basis, blocks, s, m2)
        comm = lm @ mat - blocks[(s, m2 - 2)][0] @ lm
        assert np.abs(comm).max() < tol, (s, m2)
        checked += 1
    assert checked > 10


def _lowering_t(basis: TripleBasis, blocks, sector: int, m2: int) -> np.ndarray:
    """T_- from block (sector, m2) to (sector - 1, m2) on triple states (neutron -> proton)."""
    src, dst = blocks[(sector, m2)][1], blocks[(sector - 1, m2)][1]
    pos = {g: i for i, g in enumerate(dst)}
    tm = np.zeros((len(dst), len(src)))
    for col, tri in enumerate(src):
        for slot in range(3):
            if tri[slot] < basis.nsp:
                continue
            new = list(tri)
            new[slot] -= basis.nsp
            if len(set(new)) < 3:
                continue
            inv = sum(new[i] > new[k] for i, k in itertools.combinations(range(3), 2))
            tm[pos[tuple(sorted(new))], col] += (-1) ** inv
    return tm


def test_isospin_invariance(space):
    basis, _, blocks, tol = space
    checked = 0
    for (s, m2), (mat, _) in blocks.items():
        if s == 0 or (s - 1, m2) not in blocks:
            continue
        tm = _lowering_t(basis, blocks, s, m2)
        comm = tm @ mat - blocks[(s - 1, m2)][0] @ tm
        assert np.abs(comm).max() < tol, (s, m2)
        checked += 1
    assert checked > 10


def _slots_in_file(basis: TripleBasis, tuples, states) -> int:
    """Number of (row, column) pairs of a block whose orbit triples form a file tuple."""
    keys = {(t.bra, t.ket) for t in tuples}
    keys |= {(k, b) for b, k in keys}
    orb = [tuple(sorted((basis.decode(g)[1] for g in st), reverse=True)) for st in states]
    classes = sorted(set(orb))
    cid = {c: i for i, c in enumerate(classes)}
    ok = np.array([[(a, b) in keys for b in classes] for a in classes])
    idx = np.array([cid[o] for o in orb])
    return int(ok[np.ix_(idx, idx)].sum())


def test_streaming_stats_match_dense(space):
    basis, tuples, blocks, _ = space
    st = ThreeBodyStats(basis)
    for t in tuples:
        st.add_tuple(t)
    got = st.blocks()
    assert set(got) == {(SECTORS3[s], m2) for s, m2 in blocks}
    for (s, m2), (mat, states) in blocks.items():
        b = got[(SECTORS3[s], m2)]
        a = np.abs(mat)
        assert b.lambda_full == pytest.approx(a.sum(), rel=1e-12), (s, m2)
        assert b.lambda_diag == pytest.approx(np.trace(a), rel=1e-12), (s, m2)
        assert b.nnz_full[0] == int((a > 0).sum()), (s, m2)
        assert b.nnz_ut(0) == int((np.triu(a) > 0).sum()), (s, m2)
        assert b.structural_full == _slots_in_file(basis, tuples, states), (s, m2)
        assert b.structural_full <= a.size
        assert b.vmax == pytest.approx(a.max(), rel=1e-12)


def _explicit(basis: TripleBasis, table, gb: tuple, gk: tuple) -> float:
    """<pqr|W|stu> by the Clebsch-Gordan sum of docs/spec.md 3 with sympy, element by element."""

    def slots(tri):
        st = sorted((basis.decode(g) + (g,) for g in tri), key=lambda s: (-s[1], s[3]))
        inv = sum(st[i][3] > st[k][3] for i, k in itertools.combinations(range(3), 2))
        return st, (-1) ** inv

    sb, sgn_b = slots(gb)
    sk, sgn_k = slots(gk)
    orb_b = tuple(s[1] for s in sb)
    orb_k = tuple(s[1] for s in sk)
    transposed = orb_b < orb_k
    first, second = (sk, sb) if transposed else (sb, sk)
    key = (tuple(s[1] for s in first), tuple(s[1] for s in second))
    if key not in table:  # opposite parity: no file tuple, element is zero
        return 0.0
    tup = table[key]
    jf = [basis.orbits[s[1]].j2 for s in first]
    js = [basis.orbits[s[1]].j2 for s in second]
    half = Rational(1, 2)

    def coupling(states, j, j12, t12, jj, tt):
        (_, o1, m1, _), (_, o2, m2, _), (_, o3, m3, _) = states
        t1, t2, t3 = (-half if s[0] == 0 else half for s in states)
        ang = (
            CG(
                Rational(j[0], 2),
                Rational(m1, 2),
                Rational(j[1], 2),
                Rational(m2, 2),
                j12,
                Rational(m1 + m2, 2),
            ).doit()
            * CG(
                j12,
                Rational(m1 + m2, 2),
                Rational(j[2], 2),
                Rational(m3, 2),
                Rational(jj, 2),
                Rational(m1 + m2 + m3, 2),
            ).doit()
        )
        iso = (
            CG(half, t1, half, t2, t12, t1 + t2).doit()
            * CG(t12, t1 + t2, half, t3, Rational(tt, 2), t1 + t2 + t3).doit()
        )
        return ang * iso

    total = 0.0
    entries = tuple_entries(tuple(jf), tuple(js))
    for v, (j12, t12, j45, t45, jj, tt) in zip(tup.values, entries, strict=True):
        if v != 0.0:
            total += float(
                v * coupling(first, jf, j12, t12, jj, tt) * coupling(second, js, j45, t45, jj, tt)
            )
    return sgn_b * sgn_k * total


def test_matches_explicit_sympy_sum(space):
    basis, tuples, blocks, _ = space
    table = {(t.bra, t.ket): t for t in tuples}
    rng = np.random.default_rng(7)
    done = 0
    for key in sorted(blocks):
        mat, states = blocks[key]
        nz = np.argwhere(np.abs(mat) > 0)
        zr = np.argwhere(mat == 0)
        picks = [nz[i] for i in rng.choice(len(nz), size=min(3, len(nz)), replace=False)]
        if len(zr):
            picks.append(zr[rng.integers(len(zr))])
        for r, c in picks:
            ref = _explicit(basis, table, states[r], states[c])
            assert mat[r, c] == pytest.approx(ref, abs=1e-10), (key, states[r], states[c])
            done += 1
        if done > 60:
            break
    assert done > 20


def test_batched_tensor_equals_the_block_kernel(space):
    """The padded tensor holds exactly the blocks of ``triple_blocks``, signs included."""
    basis, tuples, *_ = space
    nblocks = 0
    for tup in tuples:
        tt = tuple_tensor(basis, tup, signed=True)
        old = {(b.assign_bra, b.assign_ket, b.m2): b.x for b in triple_blocks(basis, tup)}
        new = set()
        for p in range(len(tt.pair_bra)):
            for m, m2 in enumerate(tt.m2):
                rows = np.flatnonzero(tt.ok_bra[p, m])
                cols = np.flatnonzero(tt.ok_ket[p, m])
                if len(rows) == 0 or len(cols) == 0:
                    continue
                key = (int(tt.pair_bra[p]), int(tt.pair_ket[p]), int(m2))
                new.add(key)
                assert np.allclose(tt.x[p, m][np.ix_(rows, cols)], old[key], atol=1e-13, rtol=0)
                nblocks += 1
        assert new == set(old)
    assert nblocks > 1000
