"""Two-body m-scheme expansion of J-coupled ``me2j`` elements.

The rule (docs/spec.md, section 3) is a plain Clebsch-Gordan sum of the file value, with no
extra normalisation factor, over the orbit-sorted pair (a >= b, c >= d) of the file loop. Species
(proton = 0, neutron = 1) follow the particles. This module builds, for one orbit quadruple of the
file, the dense m-scheme tensors for every sector and species assignment.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from nuham_mscheme.cg import cg_table
from nuham_mscheme.ho1b import rho_block
from nuham_mscheme.me2j import Me2jTuple
from nuham_mscheme.orbits import Orbit

PP, PN, NN = 0, 1, 2
SECTORS = ("pp", "pn", "nn")


@dataclass(frozen=True)
class QuadBlock:
    """Unique-pair-state elements <pq|V|rs> of one sector and species assignment of a quadruple.

    Attributes
    ----------
    sector : int
        PP, PN or NN.
    bra_assign, ket_assign : int
        Species order of the bra/ket pair: 0 means (proton, neutron) for (first, second orbit),
        1 means (neutron, proton). Always 0 in the pp and nn sectors.
    x : numpy.ndarray
        Elements, shape (na, nb, nc, nd), index i = m + j (see ``cg_table``). Zero where the bra or
        ket pair state is a duplicate (a == b with m_a <= m_b) or where M differs.
    mb2, mk2 : numpy.ndarray
        Twice the pair M of each bra (na, nb) and ket (nc, nd) combination.
    valid_b, valid_k : numpy.ndarray
        Boolean (na, nb), (nc, nd): the combination is a unique pair state.
    """

    sector: int
    bra_assign: int
    ket_assign: int
    x: np.ndarray
    mb2: np.ndarray
    mk2: np.ndarray
    valid_b: np.ndarray
    valid_k: np.ndarray


def _pair_masks(oa: Orbit, ob: Orbit, same_species: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (M2 array, unique-state mask) for the pair (oa, ob), shape (na, nb).

    With the same species and the same orbit the pair states (m_a, m_b) and (m_b, m_a) coincide,
    so only m_a > m_b is kept. With different species every combination is a distinct state.
    """
    ia = np.arange(oa.j2 + 1)
    ib = np.arange(ob.j2 + 1)
    m2 = (2 * ia - oa.j2)[:, None] + (2 * ib - ob.j2)[None, :]
    if same_species and oa == ob:
        valid = ia[:, None] > ib[None, :]
    else:
        valid = np.ones((oa.j2 + 1, ob.j2 + 1), dtype=bool)
    return m2, valid


@dataclass(frozen=True)
class TInt:
    """Parameters of the intrinsic kinetic energy T_int (docs/spec.md 4.2).

    Attributes
    ----------
    hw : float
        Oscillator energy hbar*omega in MeV.
    a : int
        Mass number A.
    """

    hw: float
    a: int


def _sign(e1: int, e2: int) -> int:
    return 1 if e1 > e2 else -1


def _product_tensors(
    oa: Orbit, ob: Orbit, oc: Orbit, od: Orbit, coef: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (direct, exchange) tensors of coef * (pi_1 . pi_2) in the product basis.

    direct[ia, ib, ic, id] = <a b|pi1.pi2|c d>, exchange[ia, ib, ic, id] = <a b|pi1.pi2|d c>, with
    pi1.pi2 = -s s' (rho . rho) for orbit pairs differing by one quantum (derivation 02 Eq. 7).
    """
    na, nb, nc, nd = oa.j2 + 1, ob.j2 + 1, oc.j2 + 1, od.j2 + 1
    direct = np.zeros((na, nb, nc, nd))
    exchange = np.zeros((na, nb, nc, nd))
    if abs(oa.e - oc.e) == 1 and abs(ob.e - od.e) == 1:
        r1, r2 = rho_block(oa, oc), rho_block(ob, od)
        sgn = -_sign(oa.e, oc.e) * _sign(ob.e, od.e)
        for mu in (-1, 0, 1):
            direct += sgn * (-1) ** mu * np.einsum("ik,jl->ijkl", r1[mu + 1], r2[-mu + 1])
    if abs(oa.e - od.e) == 1 and abs(ob.e - oc.e) == 1:
        r1, r2 = rho_block(oa, od), rho_block(ob, oc)
        sgn = -_sign(oa.e, od.e) * _sign(ob.e, oc.e)
        for mu in (-1, 0, 1):
            exchange += sgn * (-1) ** mu * np.einsum("il,jk->ijkl", r1[mu + 1], r2[-mu + 1])
    return coef * direct, coef * exchange


def quad_blocks(
    orbits: list[Orbit], tup: Me2jTuple, tint: TInt | None = None, include_v: bool = True
) -> Iterator[QuadBlock]:
    """Yield the m-scheme blocks of one file quadruple.

    Parameters
    ----------
    orbits : list[Orbit]
        Orbit list of the file.
    tup : Me2jTuple
        The quadruple with its (nJ, 4) values.
    tint : TInt, optional
        If given, add the two-body part of T_int, -(hw/A) sum_{i<j} pi_i . pi_j, which the file
        does not contain. Every nonzero element has a file quadruple (J, parity conserved).
    include_v : bool, optional
        If False, leave out the file's V and yield T_int alone (needs ``tint``).

    Yields
    ------
    QuadBlock
        Blocks for pp, nn and the allowed pn species assignments.
    """
    oa, ob, oc, od = (orbits[i] for i in (tup.a, tup.b, tup.c, tup.d))
    js = range(tup.jmin, tup.jmin + len(tup.values))
    cab = np.stack([cg_table(oa.j2, ob.j2, 2 * j) for j in js])
    ccd = np.stack([cg_table(oc.j2, od.j2, 2 * j) for j in js])
    mb2, valid_b = _pair_masks(oa, ob, True)
    mk2, valid_k = _pair_masks(oc, od, True)
    _, valid_bn = _pair_masks(oa, ob, False)
    _, valid_kn = _pair_masks(oc, od, False)
    v = tup.values

    same_m = mb2[:, :, None, None] == mk2[None, None, :, :]

    def tensor(w: np.ndarray, vb: np.ndarray, vk: np.ndarray) -> np.ndarray:
        # sum_J C_ab(J, M) C_cd(J, M') is not diagonal in M by itself: impose delta_{M M'}
        keep = vb[:, :, None, None] & vk[None, None] & same_m
        if not include_v:
            return np.zeros(keep.shape)
        return np.einsum("j,jab,jcd->abcd", w, cab, ccd) * keep

    t_dir = t_exc = 0.0
    if tint is not None:
        # T_int two-body = -(hw/A) pi.pi; antisymmetrised <pq|T|rs> = direct - exchange.
        t_dir, t_exc = _product_tensors(oa, ob, oc, od, -tint.hw / tint.a)

    def with_tint(
        x: np.ndarray, vb: np.ndarray, vk: np.ndarray, extra: np.ndarray | float
    ) -> np.ndarray:
        if tint is None:
            return x
        return x + extra * (vb[:, :, None, None] & vk[None, None] & same_m)

    pp = with_tint(tensor(v[:, 3], valid_b, valid_k), valid_b, valid_k, t_dir - t_exc)
    nn = with_tint(tensor(v[:, 1], valid_b, valid_k), valid_b, valid_k, t_dir - t_exc)
    yield QuadBlock(PP, 0, 0, pp, mb2, mk2, valid_b, valid_k)
    yield QuadBlock(NN, 0, 0, nn, mb2, mk2, valid_b, valid_k)
    # pn: (pn;pn) and (np;np) take the direct term, (pn;np) and (np;pn) minus the exchange term
    t_same = with_tint(
        tensor(0.5 * (v[:, 2] + v[:, 0]), valid_bn, valid_kn), valid_bn, valid_kn, t_dir
    )
    t_exch = with_tint(
        tensor(0.5 * (v[:, 2] - v[:, 0]), valid_bn, valid_kn), valid_bn, valid_kn, -t_exc
    )
    # pn pair states: (a p, b n) always; (a n, b p) only when a != b (else it repeats the first).
    # The assignment is a species label of the first/second orbit, so for a == b the second
    # assignment would be the same set of states with the orbits exchanged.
    bra_assigns = (0, 1) if tup.a != tup.b else (0,)
    ket_assigns = (0, 1) if tup.c != tup.d else (0,)
    for ab in bra_assigns:
        for ak in ket_assigns:
            yield QuadBlock(
                PN, ab, ak, t_same if ab == ak else t_exch, mb2, mk2, valid_bn, valid_kn
            )
