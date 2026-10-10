"""Streaming 1-norm, nonzero-count and max statistics of the three-body m-scheme matrix.

Convention "B" of docs/spec.md: the matrix is <pqr|W|stu> over unique triple states p<q<r,
s<t<u. The file tuples cover the lower triangle in orbit-triple order, so an off-diagonal tuple
stands for two mirror blocks and a tuple with bra == ket is a symmetric diagonal block. The
accumulation reuses the two-body bins (sector = number of neutrons, 2M).
"""

import numpy as np

from nuham_mscheme.me3j import Me3jTuple
from nuham_mscheme.mscheme3 import SECTORS3, TripleBasis, triple_blocks
from nuham_mscheme.stats2 import BlockStats, _Acc


class ThreeBodyStats:
    """Accumulator over file tuples.

    Parameters
    ----------
    basis : TripleBasis
        Single-particle ordering of the model space; fixes the range of 2M.
    """

    def __init__(self, basis: TripleBasis) -> None:
        self.basis = basis
        self.acc = _Acc(len(SECTORS3), basis.max_m2)

    def add_tuple(self, tup: Me3jTuple) -> None:
        """Add one file tuple."""
        diag_tuple = tup.bra == tup.ket
        mult = 1 if diag_tuple else 2  # an off-diagonal tuple stands for two mirror blocks
        moff = self.acc.moff
        for blk in triple_blocks(self.basis, tup):
            ax = np.abs(blk.x)
            self.acc.add(
                blk.sector, np.full(ax.size, blk.m2 + moff, dtype=np.intp), ax.ravel(), 0, mult
            )
            if diag_tuple and blk.assign_bra == blk.assign_ket:
                dx = np.diagonal(ax)
                self.acc.add(blk.sector, np.full(dx.size, blk.m2 + moff, dtype=np.intp), dx, 1, 1)

    def merge(self, other: "ThreeBodyStats") -> None:
        """Add the statistics of another accumulator, in place."""
        self.acc.merge(other.acc)

    def blocks(self) -> dict[tuple[str, int], BlockStats]:
        """Return the ``{(sector name, 2M): BlockStats}`` statistics."""
        return self.acc.blocks(SECTORS3)
