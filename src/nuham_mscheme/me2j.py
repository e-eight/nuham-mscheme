"""Reader for the Darmstadt ``me2j`` two-body file written by NuHamil and nuham-jax."""

import gzip
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nuham_mscheme.orbits import Orbit, orbit_list


@dataclass(frozen=True)
class Me2jTuple:
    """One orbit quadruple of the file with all its J values.

    Attributes
    ----------
    a, b, c, d : int
        0-based orbit indices, a >= b, c <= a, d <= (b if c == a else c).
    jmin : int
        Smallest J (true value, integer).
    values : numpy.ndarray
        Shape (nJ, 4), columns (T=0, nn, pn T=1, pp), MeV, un-normalised (Darmstadt).
    """

    a: int
    b: int
    c: int
    d: int
    jmin: int
    values: np.ndarray


def tuple_layout(orbits: list[Orbit], e2max: int) -> Iterator[tuple[int, int, int, int, int, int]]:
    """Yield (a, b, c, d, jmin, jmax) in the order of the file, J range possibly empty.

    Parameters
    ----------
    orbits : list[Orbit]
        Orbit list of the file (``orbit_list(emax)``).
    e2max : int
        e2max of the file.

    Yields
    ------
    tuple[int, int, int, int, int, int]
        Orbit indices and the J range. Tuples with jmin > jmax carry no values and are skipped.
    """
    for a, oa in enumerate(orbits):
        for b in range(a + 1):
            ob = orbits[b]
            if oa.e + ob.e > e2max:
                continue
            for c in range(a + 1):
                oc = orbits[c]
                for d in range((b if c == a else c) + 1):
                    od = orbits[d]
                    if oc.e + od.e > e2max:
                        continue
                    if (oa.l + ob.l + oc.l + od.l) % 2:
                        continue
                    jmin = max(abs(oa.j2 - ob.j2), abs(oc.j2 - od.j2)) // 2
                    jmax = min(oa.j2 + ob.j2, oc.j2 + od.j2) // 2
                    if jmin <= jmax:
                        yield a, b, c, d, jmin, jmax


def tuple_count(emax: int, e2max: int) -> int:
    """Return the number of orbit quadruples of a file with these limits."""
    return sum(1 for _ in tuple_layout(orbit_list(emax), e2max))


def value_count(emax: int, e2max: int) -> int:
    """Return the number of values a file with these limits holds (4 per (tuple, J))."""
    return 4 * sum(jmax - jmin + 1 for *_, jmin, jmax in tuple_layout(orbit_list(emax), e2max))


def read_values(path: str | Path) -> np.ndarray:
    """Read all values of a text or gzip ``me2j`` file after the one-line header.

    Parameters
    ----------
    path : str or Path
        File; gzip is detected from the magic bytes.

    Returns
    -------
    numpy.ndarray
        Flat float64 array of the values.
    """
    path = Path(path)
    with path.open("rb") as fh:
        magic = fh.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rt") as fh:
        fh.readline()
        return np.array(fh.read().split(), dtype=np.float64)


def iter_tuples(
    path: str | Path,
    emax: int,
    e2max: int,
    emax_use: int | None = None,
    e2max_use: int | None = None,
    index_range: tuple[int, int] | None = None,
) -> Iterator[Me2jTuple]:
    """Yield the orbit quadruples of a file with their values, checking the value count.

    Parameters
    ----------
    path : str or Path
        The file.
    emax, e2max : int
        emax and e2max the file was written with.
    emax_use : int, optional
        Keep only quadruples with all orbits e <= emax_use (a prefix of the file).
    e2max_use : int, optional
        Keep only quadruples with e_a + e_b and e_c + e_d <= e2max_use.
    index_range : tuple[int, int], optional
        Half-open range of positions in the file's quadruple sequence (before filtering). Used to
        split a file into fixed chunks; the whole value count is still checked.

    Yields
    ------
    Me2jTuple
        One quadruple.

    Raises
    ------
    ValueError
        If the file holds a different number of values than the loop implies.
    """
    orbits = orbit_list(emax)
    data = read_values(path)
    expected = value_count(emax, e2max)
    if data.size != expected:
        raise ValueError(f"{path}: {data.size} values, expected {expected} for ({emax}, {e2max})")
    pos = 0
    for k, (a, b, c, d, jmin, jmax) in enumerate(tuple_layout(orbits, e2max)):
        n = jmax - jmin + 1
        start = pos
        pos += 4 * n
        if index_range is not None and not index_range[0] <= k < index_range[1]:
            continue
        if emax_use is not None and orbits[a].e > emax_use:
            continue
        if e2max_use is not None and (
            orbits[a].e + orbits[b].e > e2max_use or orbits[c].e + orbits[d].e > e2max_use
        ):
            continue
        yield Me2jTuple(a, b, c, d, jmin, data[start:pos].reshape(n, 4))
