"""Dense two-body m-scheme matrices for tests and small spaces (not used at production size)."""

import numpy as np

from nuham_mscheme.me2j import Me2jTuple
from nuham_mscheme.mscheme2 import TInt, quad_blocks
from nuham_mscheme.orbits import Orbit


def sp_states(orbits: list[Orbit], emax: int) -> list[tuple[int, int, int]]:
    """Return single-particle states (species, orbit, 2m) with e <= emax.

    Order: protons first, then neutrons; within a species by orbit index then m ascending
    (docs/spec.md section 2; the orbit index orders by e, l, j).
    """
    out = []
    for t in (0, 1):
        for i, o in enumerate(orbits):
            if o.e <= emax:
                out.extend((t, i, 2 * k - o.j2) for k in range(o.j2 + 1))
    return out


def dense_matrix(
    orbits: list[Orbit], tuples: list[Me2jTuple], emax: int, tint: TInt | None = None
) -> tuple[np.ndarray, list[tuple[int, int]], dict[tuple[int, int, int], int]]:
    """Assemble the full matrix <pq|V|rs> over unique pair states p < q of the given quadruples.

    Parameters
    ----------
    orbits : list[Orbit]
        Orbit list of the file.
    tuples : list[Me2jTuple]
        File quadruples with all orbits e <= emax (a prefix of the file).
    emax : int
        Model emax.
    tint : TInt, optional
        If given, add the two-body part of T_int to V.

    Returns
    -------
    matrix : numpy.ndarray
        Symmetric (npairs, npairs).
    pairs : list[tuple[int, int]]
        Pair states as (p, q) indices into ``sp_states``, p < q.
    index : dict
        Map from single-particle state (species, orbit, 2m) to its index.
    """
    sps = sp_states(orbits, emax)
    index = {s: i for i, s in enumerate(sps)}
    pairs = [(p, q) for p in range(len(sps)) for q in range(p + 1, len(sps))]
    pidx = {pq: i for i, pq in enumerate(pairs)}
    mat = np.zeros((len(pairs), len(pairs)))

    def state(species: int, orbit: int, i: int) -> int:
        return index[(species, orbit, 2 * i - orbits[orbit].j2)]

    def pair_index(s1: int, s2: int) -> tuple[int, float]:
        return (pidx[(s1, s2)], 1.0) if s1 < s2 else (pidx[(s2, s1)], -1.0)

    for tup in tuples:
        for blk in quad_blocks(orbits, tup, tint=tint):
            oa, ob, oc, od = tup.a, tup.b, tup.c, tup.d
            ta, tb = ((0, 0), (0, 1), (1, 1))[blk.sector]
            if blk.sector == 1:
                ta, tb = (0, 1) if blk.bra_assign == 0 else (1, 0)
            tc, td = ((0, 0), (0, 1), (1, 1))[blk.sector]
            if blk.sector == 1:
                tc, td = (0, 1) if blk.ket_assign == 0 else (1, 0)
            for ia, ib, ic, id_ in zip(*np.nonzero(blk.x), strict=True):
                row, s1 = pair_index(state(ta, oa, ia), state(tb, ob, ib))
                col, s2 = pair_index(state(tc, oc, ic), state(td, od, id_))
                val = s1 * s2 * blk.x[ia, ib, ic, id_]
                mat[row, col] += val
                if (row, col) != (col, row) and not (oa == oc and ob == od):
                    mat[col, row] += val
    return mat, pairs, index
