"""Streaming 1-norm, nonzero-count and max statistics of the three-body m-scheme matrix.

Convention "B" of docs/spec.md: the matrix is <pqr|W|stu> over unique triple states p<q<r,
s<t<u. The file tuples cover the lower triangle in orbit-triple order, so an off-diagonal tuple
stands for two mirror blocks and a tuple with bra == ket is a symmetric diagonal block.

Each tuple is expanded into one padded tensor (``tuple_tensor``) and reduced with a handful of
numpy calls: the magnitudes are binned by sector, 2M and threshold interval, and the interval
histograms are turned into the "above threshold" counts and sums only when the blocks are read out.
"""

import numpy as np

from nuham_mscheme.me3j import Me3jTuple
from nuham_mscheme.mscheme3 import SECTORS3, TripleBasis, tuple_tensor
from nuham_mscheme.stats2 import TAUS, BlockStats

_TAUS = np.asarray(TAUS)
_NT = len(TAUS)


class _Hist:
    """Histograms over (sector, 2M, threshold interval), full matrix and diagonal separately."""

    def __init__(self, n_sectors: int, moff: int) -> None:
        self.n_sectors, self.moff, self.nm = n_sectors, moff, 2 * moff + 1
        shape = (2, n_sectors, self.nm, _NT + 1)
        self.cnt = np.zeros(shape, dtype=np.int64)
        self.wsum = np.zeros(shape)
        self.lam = np.zeros((2, n_sectors, self.nm))
        self.comp = np.zeros((2, n_sectors, self.nm))
        self.vmax = np.zeros((n_sectors, self.nm))

    def _kahan(self, which: int, x: np.ndarray) -> None:
        s = self.lam[which]
        t = s + x
        self.comp[which] += np.where(np.abs(s) >= np.abs(x), (s - t) + x, (x - t) + s)
        self.lam[which] = t

    def add(
        self, which: int, sector: np.ndarray, midx: np.ndarray, ax: np.ndarray, mult: int
    ) -> None:
        """Add magnitudes ``ax`` with their sector and 2M-bin index arrays (all one shape)."""
        if ax.size == 0:
            return
        size = self.n_sectors * self.nm * (_NT + 1)
        key = (sector * self.nm + midx) * (_NT + 1) + np.searchsorted(_TAUS, ax, side="left")
        cnt = np.bincount(key, minlength=size).reshape(self.n_sectors, self.nm, _NT + 1)
        wsum = np.bincount(key, weights=ax, minlength=size).reshape(cnt.shape)
        self.cnt[which] += mult * cnt
        self.wsum[which] += mult * wsum
        self._kahan(which, mult * wsum.sum(axis=-1))

    def merge(self, other: "_Hist") -> None:
        """Add another histogram in place (integer sums exact, 1-norms compensated)."""
        self.cnt += other.cnt
        self.wsum += other.wsum
        for which in (0, 1):
            self._kahan(which, other.lam[which] + other.comp[which])
        self.vmax = np.maximum(self.vmax, other.vmax)

    def blocks(self) -> dict[tuple[str, int], BlockStats]:
        """Return ``{(sector name, 2M): BlockStats}``."""

        def tail(a: np.ndarray) -> np.ndarray:
            # entries above TAUS[i] are the intervals i+1..nt
            return np.cumsum(a[..., :0:-1], axis=-1)[..., ::-1]

        nnz, above = tail(self.cnt), tail(self.wsum)
        struct = self.cnt.sum(axis=-1)
        out = {}
        for sec, name in enumerate(SECTORS3):
            for b in np.nonzero(struct[0, sec])[0]:
                out[(name, int(b) - self.moff)] = BlockStats(
                    structural_full=int(struct[0, sec, b]),
                    structural_diag=int(struct[1, sec, b]),
                    nnz_full=[int(v) for v in nnz[0, sec, b]],
                    nnz_diag=[int(v) for v in nnz[1, sec, b]],
                    lam_full=float(self.lam[0, sec, b] + self.comp[0, sec, b]),
                    lam_diag=float(self.lam[1, sec, b] + self.comp[1, sec, b]),
                    lam_above_full=[float(v) for v in above[0, sec, b]],
                    lam_above_diag=[float(v) for v in above[1, sec, b]],
                    vmax=float(self.vmax[sec, b]),
                )
        return out


class ThreeBodyStats:
    """Accumulator over file tuples.

    Parameters
    ----------
    basis : TripleBasis
        Single-particle ordering of the model space; fixes the range of 2M.
    """

    def __init__(self, basis: TripleBasis) -> None:
        self.basis = basis
        self.acc = _Hist(len(SECTORS3), basis.max_m2)

    def add_tuple(self, tup: Me3jTuple) -> None:
        """Add one file tuple."""
        diag_tuple = tup.bra == tup.ket
        mult = 1 if diag_tuple else 2  # an off-diagonal tuple stands for two mirror blocks
        tt = tuple_tensor(self.basis, tup)
        acc = self.acc
        sec = tt.pair_sector[:, None]
        midx = (tt.m2 + acc.moff)[None, :]
        valid = tt.valid
        ax = np.abs(tt.x)
        shape = ax.shape
        acc.add(
            0,
            np.broadcast_to(sec[:, :, None, None], shape)[valid],
            np.broadcast_to(midx[:, :, None, None], shape)[valid],
            ax[valid],
            mult,
        )
        top = np.where(valid, ax, 0.0).max(axis=(2, 3))
        np.maximum.at(
            acc.vmax, (np.broadcast_to(sec, top.shape), np.broadcast_to(midx, top.shape)), top
        )
        if diag_tuple:
            dp = np.nonzero(tt.pair_bra == tt.pair_ket)[0]
            dx = np.diagonal(ax[dp], axis1=2, axis2=3)
            ok = tt.ok_bra[dp]
            acc.add(
                1,
                np.broadcast_to(sec[dp][:, :, None], dx.shape)[ok],
                np.broadcast_to(midx[:, :, None], dx.shape)[ok],
                dx[ok],
                1,
            )

    def merge(self, other: "ThreeBodyStats") -> None:
        """Add the statistics of another accumulator, in place."""
        self.acc.merge(other.acc)

    def blocks(self) -> dict[tuple[str, int], BlockStats]:
        """Return the ``{(sector name, 2M): BlockStats}`` statistics."""
        return self.acc.blocks()
