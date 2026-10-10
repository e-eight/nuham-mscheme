"""Streaming 1-norm, nonzero-count and max statistics of the two-body m-scheme matrix.

Convention "B" of docs/spec.md: the matrix is <pq|V|rs> over unique pair states p<q, r<s. The file
quadruples cover the upper triangle in orbit-pair order, so an off-diagonal quadruple stands for two
mirror blocks, and a quadruple with (a, b) == (c, d) is a symmetric diagonal block.
"""

from dataclasses import dataclass, field

import numpy as np

from nuham_mscheme.me2j import Me2jTuple
from nuham_mscheme.mscheme2 import SECTORS, QuadBlock, TInt, quad_blocks
from nuham_mscheme.orbits import Orbit

TAUS = (
    0.0,
    5e-9,
    *(m * 10.0**e for e in range(-6, -1) for m in (1, 2, 5)),
    1e-1,
)
"""Nonzero thresholds, ascending: exact, file print precision (5e-9), then a 1-2-5 ladder from 1e-6
(imsrg++ read cutoff, docs/spec.md 4.3) to 0.1 MeV, to test magnitude cuts against published
counts."""

PARTS = ("v", "tint", "v+tint")
"""Statistics tracked: the file's V, the two-body T_int alone, and their sum."""

_MOFF = 40  # offset so that 2M + _MOFF indexes arrays; |2M| <= 2 * 17 for emax <= 9
_NBIN = 2 * _MOFF + 1


@dataclass
class BlockStats:
    """Statistics of one (sector, M2) block, full matrix and diagonal separately.

    Attributes
    ----------
    structural_full, structural_diag : int
        Slots allowed by M conservation, whole matrix and diagonal.
    nnz_full, nnz_diag : list[int]
        Elements with |V| > tau for each tau in ``TAUS``.
    lam_full, lam_diag : float
        Sum of |V| over the whole matrix and over the diagonal (compensated sums).
    lam_above_full, lam_above_diag : list[float]
        Sum of |V| over the elements with |V| > tau, for each tau in ``TAUS`` (plain sums, a
        diagnostic for how much 1-norm a magnitude cut would remove).
    vmax : float
        Largest |V|.
    """

    structural_full: int = 0
    structural_diag: int = 0
    nnz_full: list[int] = field(default_factory=lambda: [0] * len(TAUS))
    nnz_diag: list[int] = field(default_factory=lambda: [0] * len(TAUS))
    lam_full: float = 0.0
    lam_diag: float = 0.0
    lam_above_full: list[float] = field(default_factory=lambda: [0.0] * len(TAUS))
    lam_above_diag: list[float] = field(default_factory=lambda: [0.0] * len(TAUS))
    vmax: float = 0.0

    @property
    def lambda_full(self) -> float:
        """1-norm over the whole matrix (convention B lambda)."""
        return self.lam_full

    @property
    def lambda_diag(self) -> float:
        """1-norm of the diagonal."""
        return self.lam_diag

    @property
    def lambda_ut(self) -> float:
        """1-norm of the upper triangle including the diagonal, unweighted."""
        return 0.5 * (self.lam_full + self.lam_diag)

    def lambda_above_ut(self, i: int) -> float:
        """1-norm of the upper-triangle elements with |V| > ``TAUS[i]``."""
        return 0.5 * (self.lam_above_full[i] + self.lam_above_diag[i])

    def nnz_ut(self, i: int) -> int:
        """Nonzeros of the upper triangle including the diagonal at ``TAUS[i]``."""
        return (self.nnz_full[i] + self.nnz_diag[i]) // 2

    def structural_ut(self) -> int:
        """Structural slots of the upper triangle including the diagonal."""
        return (self.structural_full + self.structural_diag) // 2


class _Acc:
    """Vectorised accumulator over (sector, 2M) bins with Neumaier-compensated 1-norm sums."""

    def __init__(self) -> None:
        shape = (len(SECTORS), _NBIN)
        self.struct = np.zeros(shape + (2,), dtype=np.int64)  # [..., 0] full, [..., 1] diag
        self.nnz = np.zeros(shape + (2, len(TAUS)), dtype=np.int64)
        self.lam = np.zeros(shape + (2,))
        self.lam_above = np.zeros(shape + (2, len(TAUS)))
        self.comp = np.zeros(shape + (2,))
        self.vmax = np.zeros(shape)

    def _kahan(self, sector: int, bins: np.ndarray | None, which: int, x: np.ndarray) -> None:
        s = self.lam[sector, :, which]
        t = s + x
        self.comp[sector, :, which] += np.where(np.abs(s) >= np.abs(x), (s - t) + x, (x - t) + s)
        self.lam[sector, :, which] = t

    def add(self, sector: int, midx: np.ndarray, ax: np.ndarray, which: int, mult: int) -> None:
        """Add |V| values ``ax`` of one block, binned by ``midx`` (2M + offset)."""
        if ax.size == 0:
            return
        self.struct[sector, :, which] += mult * np.bincount(midx, minlength=_NBIN)
        # bin each |V| by how many thresholds lie below it, then cumulate: |V| > TAUS[i] <=> bin > i
        nt = len(TAUS)
        key = midx * (nt + 1) + np.searchsorted(np.asarray(TAUS), ax, side="left")
        size = _NBIN * (nt + 1)
        cnt = np.bincount(key, minlength=size).reshape(_NBIN, nt + 1)
        wsum = np.bincount(key, weights=ax, minlength=size).reshape(_NBIN, nt + 1)
        # tail sums over bins i+1..nt
        self.nnz[sector, :, which] += mult * np.cumsum(cnt[:, :0:-1], axis=1)[:, ::-1]
        self.lam_above[sector, :, which] += mult * np.cumsum(wsum[:, :0:-1], axis=1)[:, ::-1]
        self._kahan(sector, midx, which, mult * np.bincount(midx, weights=ax, minlength=_NBIN))
        if which == 0:
            np.maximum.at(self.vmax[sector], midx, ax)

    def merge(self, other: "_Acc") -> None:
        """Add another accumulator in place (integer sums exact, 1-norms compensated)."""
        self.struct += other.struct
        self.nnz += other.nnz
        self.lam_above += other.lam_above
        for sec in range(len(SECTORS)):
            for which in (0, 1):
                self._kahan(sec, None, which, other.lam[sec, :, which] + other.comp[sec, :, which])
        self.vmax = np.maximum(self.vmax, other.vmax)

    def blocks(self) -> dict[tuple[str, int], BlockStats]:
        """Return the accumulated statistics as ``{(sector name, 2M): BlockStats}``."""
        out = {}
        for sec, name in enumerate(SECTORS):
            for b in np.nonzero(self.struct[sec, :, 0])[0]:
                out[(name, int(b) - _MOFF)] = BlockStats(
                    structural_full=int(self.struct[sec, b, 0]),
                    structural_diag=int(self.struct[sec, b, 1]),
                    nnz_full=[int(v) for v in self.nnz[sec, b, 0]],
                    nnz_diag=[int(v) for v in self.nnz[sec, b, 1]],
                    lam_full=float(self.lam[sec, b, 0] + self.comp[sec, b, 0]),
                    lam_diag=float(self.lam[sec, b, 1] + self.comp[sec, b, 1]),
                    lam_above_full=[float(v) for v in self.lam_above[sec, b, 0]],
                    lam_above_diag=[float(v) for v in self.lam_above[sec, b, 1]],
                    vmax=float(self.vmax[sec, b]),
                )
        return out


class TwoBodyStats:
    """Accumulator over file quadruples, with V, two-body T_int and their sum as parts."""

    def __init__(self, tint: TInt | None = None) -> None:
        """Create an empty accumulator.

        Parameters
        ----------
        tint : TInt, optional
            If given, also track the two-body T_int part and the sum ``v+tint``.
        """
        self.tint = tint
        self.parts: dict[str, _Acc] = {"v": _Acc()}
        if tint is not None:
            self.parts["tint"] = _Acc()
            self.parts["v+tint"] = _Acc()

    def blocks(self, part: str = "v") -> dict[tuple[str, int], BlockStats]:
        """Return the (sector, 2M) block statistics of a part: ``v``, ``tint`` or ``v+tint``."""
        return self.parts[part].blocks()

    def merge(self, other: "TwoBodyStats") -> None:
        """Add the statistics of another accumulator with the same parts, in place."""
        for name, acc in self.parts.items():
            acc.merge(other.parts[name])

    def add_tuple(self, orbits: list[Orbit], tup: Me2jTuple) -> None:
        """Add one file quadruple.

        Parameters
        ----------
        orbits : list[Orbit]
            Orbit list of the file.
        tup : Me2jTuple
            The quadruple.
        """
        is_diag_block = tup.a == tup.c and tup.b == tup.d
        blocks_v = list(quad_blocks(orbits, tup))
        for blk in blocks_v:
            self._add_block("v", blk, blk.x, is_diag_block)
        if self.tint is None:
            return
        for blk_v, blk_t in zip(
            blocks_v, quad_blocks(orbits, tup, tint=self.tint, include_v=False), strict=True
        ):
            self._add_block("tint", blk_t, blk_t.x, is_diag_block)
            self._add_block("v+tint", blk_t, blk_v.x + blk_t.x, is_diag_block)

    def _add_block(self, part: str, blk: QuadBlock, x: np.ndarray, is_diag_block: bool) -> None:
        acc = self.parts[part]
        mb2 = blk.mb2[:, :, None, None]
        mk2 = blk.mk2[None, None, :, :]
        allowed = (mb2 == mk2) & blk.valid_b[:, :, None, None] & blk.valid_k[None, None]
        mult = 1 if is_diag_block else 2  # an off-diagonal quadruple stands for two mirror blocks
        midx = (np.broadcast_to(mb2, allowed.shape)[allowed] + _MOFF).astype(np.intp)
        acc.add(blk.sector, midx, np.abs(x[allowed]), 0, mult)
        if is_diag_block and blk.bra_assign == blk.ket_assign:
            dx = np.abs(np.einsum("abab->ab", x))[blk.valid_b]
            acc.add(blk.sector, (blk.mb2[blk.valid_b] + _MOFF).astype(np.intp), dx, 1, 1)
