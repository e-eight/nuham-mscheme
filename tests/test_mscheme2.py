import numpy as np
import pytest

from nuham_mscheme.dense2 import dense_matrix, sp_states
from nuham_mscheme.me2j import iter_tuples, value_count
from nuham_mscheme.orbits import orbit_list
from nuham_mscheme.stats2 import TwoBodyStats


def test_value_counts_match_files(e4_file, e6_file):
    assert value_count(4, 8) == 27336
    assert value_count(8, 16) == 2496152
    assert sum(1 for _ in iter_tuples(e4_file, 4, 8)) > 0
    assert sum(1 for _ in iter_tuples(e6_file, 6, 12)) > 0


def _lowering(orbits, sps, pairs):
    """J_- on pair states |p q> = a+_p a+_q |0>, p < q."""
    index = {s: i for i, s in enumerate(sps)}
    pidx = {pq: i for i, pq in enumerate(pairs)}
    L = np.zeros((len(pairs), len(pairs)))
    for col, (p, q) in enumerate(pairs):
        for slot in (0, 1):
            t, orb, m2 = sps[(p, q)[slot]]
            j2 = orbits[orb].j2
            if m2 - 2 < -j2:
                continue
            amp = np.sqrt((j2 + m2) / 2 * (j2 - m2 + 2) / 2)
            new = index[(t, orb, m2 - 2)]
            pair = [p, q]
            pair[slot] = new
            if pair[0] == pair[1]:
                continue
            sign = 1.0
            if pair[0] > pair[1]:
                pair.reverse()
                sign = -1.0
            L[pidx[(pair[0], pair[1])], col] += sign * amp
    return L


@pytest.fixture(scope="module")
def small(e4_file):
    orbits = orbit_list(4)
    emax = 2
    tuples = [t for t in iter_tuples(e4_file, 4, 8, emax_use=emax)]
    mat, pairs, index = dense_matrix(orbits, tuples, emax)
    return orbits, emax, mat, pairs


def test_hermitian(small):
    _, _, mat, _ = small
    assert np.abs(mat).max() > 1.0
    assert np.abs(mat - mat.T).max() < 1e-12


def test_rotational_invariance(small):
    orbits, emax, mat, pairs = small
    sps = sp_states(orbits, emax)
    L = _lowering(orbits, sps, pairs)
    comm = L @ mat - mat @ L
    assert np.abs(comm).max() < 1e-9 * np.abs(mat).max()


def test_charge_symmetry_pp_nn_structure(small):
    # pp and nn blocks are separate files columns; their M-structure must obey M -> -M symmetry
    orbits, emax, mat, pairs = small
    sps = sp_states(orbits, emax)
    m_of = np.array([sps[p][2] + sps[q][2] for p, q in pairs])
    for m in range(0, 5):
        sel_p = np.where(m_of == m)[0]
        sel_m = np.where(m_of == -m)[0]
        assert np.abs(mat[np.ix_(sel_p, sel_p)]).sum() == pytest.approx(
            np.abs(mat[np.ix_(sel_m, sel_m)]).sum(), rel=1e-12
        )


def test_streaming_stats_match_dense(small, e4_file):
    orbits, emax, mat, pairs = small
    sps = sp_states(orbits, emax)
    st = TwoBodyStats()
    for t in iter_tuples(e4_file, 4, 8, emax_use=emax):
        st.add_tuple(orbits, t)
    lam = sum(b.lambda_full for b in st.blocks("v").values())
    assert lam == pytest.approx(np.abs(mat).sum(), rel=1e-12)
    nnz = sum(b.nnz_full[0] for b in st.blocks("v").values())
    assert nnz == int((mat != 0).sum())
    diag = sum(b.lambda_diag for b in st.blocks("v").values())
    assert diag == pytest.approx(np.abs(np.diag(mat)).sum(), rel=1e-12)
    # per-sector, per-M block
    species = np.array([[sps[p][0], sps[q][0]] for p, q in pairs])
    m_of = np.array([sps[p][2] + sps[q][2] for p, q in pairs])
    sector = species.sum(axis=1)  # 0 pp, 1 pn, 2 nn
    names = ("pp", "pn", "nn")
    for (name, m2), b in st.blocks("v").items():
        sel = np.where((sector == names.index(name)) & (m_of == m2))[0]
        sub = mat[np.ix_(sel, sel)]
        assert b.lambda_full == pytest.approx(np.abs(sub).sum(), rel=1e-12), (name, m2)
        assert b.nnz_full[0] == int((sub != 0).sum()), (name, m2)
        assert b.vmax == pytest.approx(np.abs(sub).max(), rel=1e-12)


def test_tint_two_body_is_hermitian_scalar(e4_file):
    from nuham_mscheme.mscheme2 import TInt

    orbits = orbit_list(4)
    emax = 2
    tuples = list(iter_tuples(e4_file, 4, 8, emax_use=emax))
    v, pairs, _ = dense_matrix(orbits, tuples, emax)
    vt, _, _ = dense_matrix(orbits, tuples, emax, tint=TInt(16.0, 16))
    t = vt - v
    assert np.abs(t).max() > 0.1
    assert np.abs(t - t.T).max() < 1e-12
    L = _lowering(orbits, sp_states(orbits, emax), pairs)
    assert np.abs(L @ t - t @ L).max() < 1e-9 * np.abs(t).max()
