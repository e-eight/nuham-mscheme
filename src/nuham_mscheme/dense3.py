"""Dense three-body m-scheme matrices of small spaces, for tests (not used at production size)."""

from collections.abc import Iterable

import numpy as np

from nuham_mscheme.me3j import Me3jTuple
from nuham_mscheme.mscheme3 import TripleBasis, triple_blocks


def dense_blocks(
    basis: TripleBasis, tuples: Iterable[Me3jTuple]
) -> dict[tuple[int, int], tuple[np.ndarray, list[tuple[int, int, int]]]]:
    """Assemble the matrices <pqr|W|stu> per (sector, 2M) block from file tuples.

    Parameters
    ----------
    basis : TripleBasis
        Single-particle ordering.
    tuples : iterable of Me3jTuple
        File tuples of a model space (a prefix of the file, so that the bra and ket triples of
        every state pair are present).

    Returns
    -------
    dict
        ``{(sector, 2M): (matrix, states)}`` with ``states[i]`` the sorted global single-particle
        indices of row/column i. Each matrix element is written by exactly one tuple (or its
        mirror image); a second write raises ``RuntimeError``.
    """
    blocks = [(t, b) for t in tuples for b in triple_blocks(basis, t)]
    states: dict[tuple[int, int], dict[tuple[int, int, int], int]] = {}
    for _, b in blocks:
        d = states.setdefault((b.sector, b.m2), {})
        for g in (*map(tuple, b.bra_g.tolist()), *map(tuple, b.ket_g.tolist())):
            d.setdefault(g, 0)
    for d in states.values():
        for i, g in enumerate(sorted(d)):
            d[g] = i
    mats = {k: np.zeros((len(d), len(d))) for k, d in states.items()}
    written = {k: np.zeros((len(d), len(d)), dtype=bool) for k, d in states.items()}
    for tup, b in blocks:
        d = states[(b.sector, b.m2)]
        r = np.array([d[tuple(g)] for g in b.bra_g.tolist()])
        c = np.array([d[tuple(g)] for g in b.ket_g.tolist()])
        pairs = [(r, c, b.x)]
        if tup.bra != tup.ket:
            pairs.append((c, r, b.x.T))
        for rr, cc, xx in pairs:
            if written[(b.sector, b.m2)][np.ix_(rr, cc)].any():
                raise RuntimeError(f"element written twice, tuple {tup.bra}, {tup.ket}")
            written[(b.sector, b.m2)][np.ix_(rr, cc)] = True
            mats[(b.sector, b.m2)][np.ix_(rr, cc)] = xx
    return {k: (mats[k], sorted(states[k], key=states[k].__getitem__)) for k in mats}
