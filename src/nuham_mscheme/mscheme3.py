"""Three-body m-scheme expansion of J-coupled ``me3j`` elements.

The rule (docs/spec.md, section 3) is a plain Clebsch-Gordan sum of the file value, with no extra
normalisation factor. For the product state |a m_a t_a, b m_b t_b, c m_c t_c> in the file's slot
order (orbit index descending) the element between the antisymmetrised, normalised states is

    <A|W|A'> = sum_{J12 T12 J45 T45 J T} CG(a b J12) CG(J12 c J) CG(T) ... W_file CG ... ,

and a unique triple state p < q < r equals such a state times the sign of the permutation that sorts
the slots. When two slots hold the same orbit, only the slot assignment with increasing single-
particle index is kept, so every unique state appears once.

The per-tuple assembly is X = C_bra . U . C_ket^T with the spatial and isospin coefficients kept
apart: for a species assignment pair the file values are first contracted with the isospin
coefficients into U_J[j12, j45], then sandwiched between spatial coefficient matrices of each
M block.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

import numpy as np

from nuham_mscheme.cg import cg_table, clebsch_gordan
from nuham_mscheme.me3j import Me3jTuple, tuple_entries
from nuham_mscheme.orbits import Orbit

SECTORS3 = ("ppp", "ppn", "pnn", "nnn")
"""Sector index = number of neutrons. Protons have t_z = -1/2, neutrons +1/2."""

PROTON, NEUTRON = 0, 1
N_ASSIGN = 8
"""Species assignments of the three slots, code = 4 t_a + 2 t_b + t_c with t = 0 (p), 1 (n)."""


@dataclass(frozen=True)
class SlotStates:
    """Product states of one orbit triple and species assignment, flattened over (m_a, m_b, m_c).

    Attributes
    ----------
    m2 : numpy.ndarray
        Twice the total M of each product state.
    ok : numpy.ndarray
        Boolean: the state is the canonical slot assignment of a unique triple state.
    sign : numpy.ndarray
        Sign of the permutation that sorts the slots into increasing single-particle index.
    g : numpy.ndarray
        Shape (n, 3): the sorted global single-particle indices (the unique triple p < q < r).
    """

    m2: np.ndarray
    ok: np.ndarray
    sign: np.ndarray
    g: np.ndarray


class TripleBasis:
    """Single-particle ordering of a model space and the per-triple state tables.

    The global index of a single-particle state is ``species * nsp + off[orbit] + (m + j)``: protons
    first, then neutrons; inside a species by orbit index and m ascending (docs/spec.md section 2).

    Parameters
    ----------
    orbits : list[Orbit]
        Orbit list of the model space (a prefix of the file's list).
    """

    def __init__(self, orbits: list[Orbit]) -> None:
        self.orbits = orbits
        self.off = np.concatenate([[0], np.cumsum([o.j2 + 1 for o in orbits])]).astype(np.int64)
        self.nsp = int(self.off[-1])
        self.max_m2 = 3 * max(o.j2 for o in orbits)
        self._cache: dict[tuple[tuple[int, int, int], int], SlotStates] = {}
        self._rows: dict[tuple[int, int, int], TripleRows] = {}

    def decode(self, g: int) -> tuple[int, int, int]:
        """Return ``(species, orbit index, 2m)`` of a global single-particle index."""
        species, rest = divmod(g, self.nsp)
        orbit = int(np.searchsorted(self.off, rest, side="right")) - 1
        return species, orbit, 2 * (rest - int(self.off[orbit])) - self.orbits[orbit].j2

    def triple_rows(self, triple: tuple[int, int, int]) -> "TripleRows":
        """Return the M-grouped, padded layout of ``triple`` (cached)."""
        if triple not in self._rows:
            m2 = self.slot_states(triple, 0).m2
            m_max = sum(self.orbits[i].j2 for i in triple)
            slot = (m2 + m_max) // 2
            order = np.argsort(slot, kind="stable")
            counts = np.bincount(slot, minlength=m_max + 1)
            pos = np.arange(len(m2)) - (np.cumsum(counts) - counts)[slot[order]]
            rows = np.full((m_max + 1, int(counts.max())), -1, dtype=np.int64)
            rows[slot[order], pos] = order
            ok = np.stack(
                [np.append(self.slot_states(triple, a).ok, False)[rows] for a in range(N_ASSIGN)]
            )
            self._rows[triple] = TripleRows(m_max, rows, ok)
        return self._rows[triple]

    def slot_states(self, triple: tuple[int, int, int], assign: int) -> SlotStates:
        """Return the product states of ``triple`` (``a >= b >= c``) with species ``assign``."""
        key = (triple, assign)
        if key not in self._cache:
            orbs = [self.orbits[i] for i in triple]
            species = [(assign >> 2) & 1, (assign >> 1) & 1, assign & 1]
            idx = np.meshgrid(*(np.arange(o.j2 + 1) for o in orbs), indexing="ij")
            g = [
                species[k] * self.nsp + int(self.off[triple[k]]) + idx[k].ravel() for k in range(3)
            ]
            ok = np.ones(g[0].shape, dtype=bool)
            if triple[0] == triple[1]:
                ok &= g[0] < g[1]
            if triple[1] == triple[2]:
                ok &= g[1] < g[2]
            inversions = (g[0] > g[1]).astype(int) + (g[0] > g[2]) + (g[1] > g[2])
            self._cache[key] = SlotStates(
                m2=np.add.reduce([2 * idx[k].ravel() - orbs[k].j2 for k in range(3)]),
                ok=ok,
                sign=1 - 2 * (inversions % 2),
                g=np.sort(np.stack(g, axis=-1), axis=-1),
            )
        return self._cache[key]


@dataclass(frozen=True)
class TripleRows:
    """The product states of an orbit triple grouped by total M, padded to a rectangle.

    Attributes
    ----------
    m_max : int
        Largest 2M of the triple, ``j_a + j_b + j_c`` (doubled j values); the 2M of row ``s`` is
        ``2 s - m_max``.
    rows : numpy.ndarray
        Shape (m_max + 1, width): flat product indices of each M, -1 where padded.
    ok : numpy.ndarray
        Shape (8, m_max + 1, width), bool: the slot is the canonical assignment of a unique state,
        for each species assignment; False on padding.
    """

    m_max: int
    rows: np.ndarray
    ok: np.ndarray


@dataclass(frozen=True)
class TripleBlock:
    """Elements <pqr|W|stu> of one species assignment pair and one M of a file tuple.

    Attributes
    ----------
    sector : int
        Number of neutrons (0..3).
    m2 : int
        Twice the total M of bra and ket.
    assign_bra, assign_ket : int
        Species assignment codes of the slots (see ``N_ASSIGN``).
    x : numpy.ndarray
        Shape (n_bra, n_ket): the elements, with the sign of each state relative to its sorted
        triple p < q < r.
    bra_g, ket_g : numpy.ndarray
        Shape (n, 3): sorted global single-particle indices of each row / column state.
    bra_sign, ket_sign : numpy.ndarray
        Signs already applied to ``x``: the element of the slot-ordered product states times
        ``bra_sign * ket_sign`` is the element of the sorted states.
    """

    sector: int
    m2: int
    assign_bra: int
    assign_ket: int
    x: np.ndarray
    bra_g: np.ndarray
    ket_g: np.ndarray
    bra_sign: np.ndarray
    ket_sign: np.ndarray


@cache
def _isospin() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(iso, tz2)``.

    ``iso[assign, t12, iT]`` = <1/2 t_a 1/2 t_b | t12 tz12> <t12 tz12 1/2 t_c | T Tz>, with iT = 0
    for T = 1/2 and 1 for T = 3/2; ``tz2[assign]`` is twice the total t_z.
    """
    iso = np.zeros((N_ASSIGN, 2, 2))
    tz2 = np.zeros(N_ASSIGN, dtype=int)
    for assign in range(N_ASSIGN):
        z = [2 * ((assign >> s) & 1) - 1 for s in (2, 1, 0)]  # doubled t_z, proton -1
        tz2[assign] = sum(z)
        for t12 in (0, 1):
            first = clebsch_gordan(1, z[0], 1, z[1], 2 * t12, z[0] + z[1])
            for it, t2 in enumerate((1, 3)):
                iso[assign, t12, it] = first * clebsch_gordan(
                    2 * t12, z[0] + z[1], 1, z[2], t2, tz2[assign]
                )
    return iso, tz2


@cache
def _spatial(j123: tuple[int, int, int], j12s: tuple[int, ...], js: tuple[int, ...]) -> np.ndarray:
    """Return the spatial coupling coefficients, shape (n_a, n_b, n_c, len(j12s), len(js)).

    ``out[ia, ib, ic, x, y]`` = sum over m12 of <ja ma jb mb | j12 m12> <j12 m12 jc mc | J M> with
    ``j12 = j12s[x]`` and doubled ``J = js[y]``, indices ``i = m + j``.
    """
    ja, jb, jc = j123
    m12 = (2 * np.arange(ja + 1) - ja)[:, None] + (2 * np.arange(jb + 1) - jb)[None, :]
    out = np.zeros((ja + 1, jb + 1, jc + 1, len(j12s), len(js)))
    for x, j12 in enumerate(j12s):
        inside = np.abs(m12) <= 2 * j12
        pos = np.where(inside, m12 // 2 + j12, 0)
        cab = cg_table(ja, jb, 2 * j12) * inside
        for y, jj in enumerate(js):
            out[:, :, :, x, y] = cab[:, :, None] * cg_table(2 * j12, jc, jj)[pos]
    out.setflags(write=False)
    return out


@cache
def _slots(
    j123: tuple[int, int, int], j456: tuple[int, int, int]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Return ``(j12s, j45s, js, index)`` for a pair of j classes.

    ``index`` has shape (n, 6): for each file value of the tuple its position (j12, t12, j45, t45,
    J, iT) in the dense array ``V[j12, t12, j45, t45, J, iT]``.
    """
    j12s = tuple(range(abs(j123[0] - j123[1]) // 2, (j123[0] + j123[1]) // 2 + 1))
    j45s = tuple(range(abs(j456[0] - j456[1]) // 2, (j456[0] + j456[1]) // 2 + 1))
    entries = list(tuple_entries(j123, j456))
    js = tuple(sorted({e[4] for e in entries}))
    index = np.array(
        [
            (j12s.index(e[0]), e[1], j45s.index(e[2]), e[3], js.index(e[4]), (e[5] - 1) // 2)
            for e in entries
        ],
        dtype=np.intp,
    ).reshape(-1, 6)
    return j12s, j45s, js, index


def triple_blocks(basis: TripleBasis, tup: Me3jTuple) -> Iterator[TripleBlock]:
    """Yield the m-scheme blocks of one file tuple.

    Parameters
    ----------
    basis : TripleBasis
        Single-particle ordering; its orbit list must be a prefix of the file's.
    tup : Me3jTuple
        The pair of orbit triples with its values.

    Yields
    ------
    TripleBlock
        One block for every pair of species assignments with equal total t_z and every M common
        to bra and ket. A tuple with bra == ket yields both ``(i, k)`` and ``(k, i)`` assignment
        pairs, so its blocks tile the whole square; an off-diagonal tuple stands for its mirror
        image as well.
    """
    orbs = basis.orbits
    j123 = (orbs[tup.bra[0]].j2, orbs[tup.bra[1]].j2, orbs[tup.bra[2]].j2)
    j456 = (orbs[tup.ket[0]].j2, orbs[tup.ket[1]].j2, orbs[tup.ket[2]].j2)
    j12s, j45s, js, index = _slots(j123, j456)
    if len(index) != len(tup.values):
        raise ValueError(
            f"tuple {tup.bra}, {tup.ket}: {len(tup.values)} values, {len(index)} slots"
        )
    vt = np.zeros((len(j12s), 2, len(j45s), 2, len(js), 2))
    vt[tuple(index.T)] = tup.values
    spb = _spatial(j123, j12s, js).reshape(-1, len(j12s), len(js))
    spk = _spatial(j456, j45s, js).reshape(-1, len(j45s), len(js))
    iso, tz2 = _isospin()
    for ab in range(N_ASSIGN):
        rb = basis.slot_states(tup.bra, ab)
        if not rb.ok.any():
            continue
        for ak in range(N_ASSIGN):
            if tz2[ab] != tz2[ak]:
                continue
            rk = basis.slot_states(tup.ket, ak)
            common = np.intersect1d(rb.m2[rb.ok], rk.m2[rk.ok])
            if common.size == 0:
                continue
            u = np.einsum("xT,yT,ixkyJT->ikJ", iso[ab], iso[ak], vt)
            for m2 in common:
                rows = np.flatnonzero(rb.ok & (rb.m2 == m2))
                cols = np.flatnonzero(rk.ok & (rk.m2 == m2))
                x = np.zeros((len(rows), len(cols)))
                for y, jj in enumerate(js):
                    if jj >= abs(m2):
                        x += (spb[rows, :, y] @ u[:, :, y]) @ spk[cols, :, y].T
                x *= rb.sign[rows][:, None] * rk.sign[cols][None, :]
                yield TripleBlock(
                    sector=bin(ab).count("1"),
                    m2=int(m2),
                    assign_bra=ab,
                    assign_ket=ak,
                    x=x,
                    bra_g=rb.g[rows],
                    ket_g=rk.g[cols],
                    bra_sign=rb.sign[rows],
                    ket_sign=rk.sign[cols],
                )


def _pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the species-assignment pairs with equal total t_z: (bra, ket, sector) arrays."""
    _, tz2 = _isospin()
    b, k = np.nonzero(tz2[:, None] == tz2[None, :])
    sector = np.array([bin(a).count("1") for a in b])
    return b, k, sector


@cache
def _spatial_pad(
    j123: tuple[int, int, int], j12s: tuple[int, ...], js: tuple[int, ...]
) -> np.ndarray:
    """Return ``_spatial`` as (n_J, n_product + 1, n_j12), with a zero row appended for padding."""
    sp = _spatial(j123, j12s, js)
    flat = sp.reshape(-1, len(j12s), len(js)).transpose(2, 0, 1)
    out = np.concatenate([flat, np.zeros((len(js), 1, len(j12s)))], axis=1)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class TupleTensor:
    """All blocks of one file tuple as one padded tensor.

    Attributes
    ----------
    x : numpy.ndarray
        Shape (n_pair, n_M, width_bra, width_ket): the elements for every species-assignment pair
        and every M common to bra and ket. Entries on padding or non-canonical slots are not
        meaningful; use ``valid``.
    ok_bra, ok_ket : numpy.ndarray
        Shape (n_pair, n_M, width), bool: the row / column is a unique state.
    m2 : numpy.ndarray
        Twice M of each of the n_M groups.
    pair_bra, pair_ket, pair_sector : numpy.ndarray
        Species assignments and sector (number of neutrons) of each pair.
    rows_bra, rows_ket : numpy.ndarray
        Shape (n_M, width): flat product index of each row / column, -1 on padding.
    """

    x: np.ndarray
    ok_bra: np.ndarray
    ok_ket: np.ndarray
    m2: np.ndarray
    pair_bra: np.ndarray
    pair_ket: np.ndarray
    pair_sector: np.ndarray
    rows_bra: np.ndarray
    rows_ket: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of the meaningful elements, shape of ``x``."""
        return self.ok_bra[:, :, :, None] & self.ok_ket[:, :, None, :]


def tuple_tensor(basis: TripleBasis, tup: Me3jTuple, signed: bool = False) -> TupleTensor:
    """Expand one file tuple into a padded tensor over (pair, M, row, column).

    Same arithmetic as ``triple_blocks`` with the Python loops over assignment pairs and M
    replaced by batched matrix products, so numpy is called once per tuple instead of once per
    block.

    Parameters
    ----------
    basis : TripleBasis
        Single-particle ordering.
    tup : Me3jTuple
        The pair of orbit triples with its values.
    signed : bool
        If True multiply by the sorting signs (the elements equal those of ``triple_blocks``); the
        statistics need only magnitudes and leave this off.
    """
    orbs = basis.orbits
    j123 = (orbs[tup.bra[0]].j2, orbs[tup.bra[1]].j2, orbs[tup.bra[2]].j2)
    j456 = (orbs[tup.ket[0]].j2, orbs[tup.ket[1]].j2, orbs[tup.ket[2]].j2)
    j12s, j45s, js, index = _slots(j123, j456)
    if len(index) != len(tup.values):
        raise ValueError(
            f"tuple {tup.bra}, {tup.ket}: {len(tup.values)} values, {len(index)} slots"
        )
    vt = np.zeros((len(j12s), 2, len(j45s), 2, len(js), 2))
    vt[tuple(index.T)] = tup.values
    pb, pk, psec = _pairs()
    iso, _ = _isospin()
    u = np.einsum("pxT,pyT,ixkyJT->pJik", iso[pb], iso[pk], vt)  # (pair, J, j12, j45)
    rb, rk = basis.triple_rows(tup.bra), basis.triple_rows(tup.ket)
    m_c = min(rb.m_max, rk.m_max)
    ob, ok_ = (rb.m_max - m_c) // 2, (rk.m_max - m_c) // 2
    n_m = m_c + 1
    rows_b = rb.rows[ob : ob + n_m]
    rows_k = rk.rows[ok_ : ok_ + n_m]
    a = _spatial_pad(j123, j12s, js)[:, rows_b]  # (J, M, rows, j12)
    bt = _spatial_pad(j456, j45s, js)[:, rows_k].transpose(0, 1, 3, 2)  # (J, M, j45, cols)
    x = np.zeros((len(pb), n_m, rows_b.shape[1], rows_k.shape[1]))
    for y in range(len(js)):
        x += (a[y][None] @ u[:, y, None]) @ bt[y][None]
    ok_b = rb.ok[:, ob : ob + n_m][pb]
    ok_k = rk.ok[:, ok_ : ok_ + n_m][pk]
    if signed:
        sb = np.stack([basis.slot_states(tup.bra, q).sign for q in range(N_ASSIGN)])
        sk = np.stack([basis.slot_states(tup.ket, q).sign for q in range(N_ASSIGN)])
        sgn_b = np.where(rows_b[None] >= 0, sb[:, rows_b], 0)[pb]
        sgn_k = np.where(rows_k[None] >= 0, sk[:, rows_k], 0)[pk]
        x *= sgn_b[:, :, :, None] * sgn_k[:, :, None, :]
    return TupleTensor(x, ok_b, ok_k, 2 * np.arange(n_m) - m_c, pb, pk, psec, rows_b, rows_k)
