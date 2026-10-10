"""One-body part of T_int in the m-scheme and its statistics."""

from nuham_mscheme.ho1b import kinetic_one_body
from nuham_mscheme.orbits import Orbit
from nuham_mscheme.stats2 import TAUS, BlockStats


def tint_one_body(o1: Orbit, o2: Orbit, hw: float, a: int) -> float:
    """Return the m-diagonal one-body element of T_int, (1 - 1/A) p^2/2m, in MeV.

    Parameters
    ----------
    o1, o2 : Orbit
        Orbits (same l and j are required for a nonzero value).
    hw : float
        hbar*omega in MeV.
    a : int
        Mass number A.
    """
    return (1.0 - 1.0 / a) * kinetic_one_body(o1, o2, hw)


def one_body_stats(orbits: list[Orbit], hw: float, a: int) -> dict[tuple[str, int], BlockStats]:
    """Return statistics of the one-body T_int matrix per (species, 2m) block.

    The matrix is diagonal in species and m, tridiagonal in n for fixed (l, j). ``lambda_full`` is
    the sum of |t_pq| over all p, q (both triangles), diagonal separately, as in the two-body
    statistics (docs/spec.md 4.2).

    Parameters
    ----------
    orbits : list[Orbit]
        Orbits of the model space (e <= emax).
    hw : float
        hbar*omega in MeV.
    a : int
        Mass number A.

    Returns
    -------
    dict
        Keys (``"p"`` or ``"n"``, 2m). Both species give identical blocks.
    """
    out: dict[tuple[str, int], BlockStats] = {}
    for species in ("p", "n"):
        for o1 in orbits:
            for o2 in orbits:
                if o1.l != o2.l or o1.j2 != o2.j2:
                    continue
                t = abs(tint_one_body(o1, o2, hw, a))
                for i in range(o1.j2 + 1):
                    st = out.setdefault((species, 2 * i - o1.j2), BlockStats())
                    st.structural_full += 1
                    for k, tau in enumerate(TAUS):
                        st.nnz_full[k] += int(t > tau)
                        st.lam_above_full[k] += t * (t > tau)
                    if o1 is o2 or o1 == o2:
                        st.structural_diag += 1
                        for k, tau in enumerate(TAUS):
                            st.nnz_diag[k] += int(t > tau)
                            st.lam_above_diag[k] += t * (t > tau)
                    st.lam_full += t
                    if o1 == o2:
                        st.lam_diag += t
                    st.vmax = max(st.vmax, t)
    return out
