"""Single-particle orbit tables, in the order used by NuHamil and imsrg++ files."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Orbit:
    """A spherical HO orbit (n, l, j) with doubled j.

    Attributes
    ----------
    n, l : int
        Radial and orbital quantum numbers.
    j2 : int
        Twice the total angular momentum.
    """

    n: int
    l: int
    j2: int

    @property
    def e(self) -> int:
        """Oscillator quanta e = 2n + l."""
        return 2 * self.n + self.l

    @property
    def parity(self) -> int:
        """Parity (-1)**l."""
        return -1 if self.l % 2 else 1


def orbit_list(emax: int, lmax: int | None = None) -> list[Orbit]:
    """Return the orbits for e = 0..emax in file order.

    The loop is e, then l (same parity as e, ascending), then j = l - 1/2 before l + 1/2
    (j = 1/2 only for l = 0). This is the NuHamil/imsrg++ "remap" list.

    Parameters
    ----------
    emax : int
        Largest e = 2n + l.
    lmax : int, optional
        Largest l; defaults to emax.

    Returns
    -------
    list[Orbit]
        The orbits. Index in the list is the 0-based orbit index used in the files.
    """
    lmax = emax if lmax is None else lmax
    out: list[Orbit] = []
    for e in range(emax + 1):
        for l in range(e % 2, min(e, lmax) + 1, 2):
            n = (e - l) // 2
            if l > 0:
                out.append(Orbit(n, l, 2 * l - 1))
            out.append(Orbit(n, l, 2 * l + 1))
    return out
