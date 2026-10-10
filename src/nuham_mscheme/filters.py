"""Model-space cutoffs matched to an NCSM Nmax space (derivation 03 of qchem-to-nuclear, Eq. 8).

The orbitals, pairs and triples that occur in the Nmax space of a nucleus with Fermi shell ``e_F``
are exactly those with ``e <= e_F + N``, ``e_a + e_b <= 2 e_F + N`` and, for triples,
``e_a + e_b + e_c <= 3 e_F + N`` with every pair inside the triple also passing the pair cut.
So the "Nmax pair-energy cut" is the ordinary ``emax_use``/``e2max_use`` prefix filter with these
values, and needs no separate code path in the readers.
"""


def matched_cutoffs(e_fermi: int, nmax: int) -> tuple[int, int, int]:
    """Return ``(emax, e2max, e3max)`` matched to ``nmax`` for Fermi shell ``e_fermi``.

    Parameters
    ----------
    e_fermi : int
        Energy ``e = 2n + l`` of the highest occupied shell (0 for 4He, 1 for 16O).
    nmax : int
        Excitation quanta above the lowest configuration.

    Returns
    -------
    tuple[int, int, int]
        ``(e_F + N, 2 e_F + N, 3 e_F + N)``. For triples the pair cut ``e2max`` must also hold.
    """
    if e_fermi < 0 or nmax < 0:
        raise ValueError("e_fermi and nmax must be non-negative")
    return e_fermi + nmax, 2 * e_fermi + nmax, 3 * e_fermi + nmax
