"""Layout of ``me3j`` files and an offset-based filter that extracts a smaller space.

The file is a header line followed by values in Fortran ``10f16.8`` (ten fields of 16 characters
per line, 161 bytes with the newline). The tuple loop and the layout are fixed by docs/spec.md 1.2
(nuham-jax spec-5d 6.2, 6.3). Because the number of values of an orbit-triple pair depends only on
the 2j values, the offset of every tuple follows from a small count table, so a smaller
(emax, e2max, e3max) space can be cut out of a large file by seeking, without parsing the skipped
values. The kept 16-character fields are copied verbatim.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Protocol

import numpy as np

from nuham_mscheme.orbits import Orbit, orbit_list

try:  # ISA-L inflates several times faster than zlib; it is an optional dependency ("fast")
    from isal import igzip as _gz
except ImportError:
    _gz = gzip

FIELD = 16
PER_LINE = 10
LINE_BYTES = FIELD * PER_LINE + 1
ISO_BLOCK = 5
"""Isospin entries per (J12, J45, 2J): (T12,T45,2T) = (0,0,1), (0,1,1), (1,0,1), (1,1,1), (1,1,3)"""


def triples(orbits: list[Orbit], e2max: int, e3max: int) -> list[tuple[int, int, int]]:
    """Return the orbit triples ``(a, b, c)``, ``a >= b >= c``, of a file in file order.

    Parameters
    ----------
    orbits : list[Orbit]
        Orbit list (e up to the file's emax, or a prefix of it).
    e2max, e3max : int
        Limits of the file: every pair ``e_i + e_j <= e2max`` and ``e_a + e_b + e_c <= e3max``.

    Returns
    -------
    list[tuple[int, int, int]]
        Lexicographically ascending, which is the order of the writer's nested loops.
    """
    out = []
    for a, oa in enumerate(orbits):
        for b in range(a + 1):
            if oa.e + orbits[b].e > e2max:
                continue
            for c in range(b + 1):
                ea, eb, ec = oa.e, orbits[b].e, orbits[c].e
                if ea + ec > e2max or eb + ec > e2max or ea + eb + ec > e3max:
                    continue
                out.append((a, b, c))
    return out


@cache
def pair_count(j123: tuple[int, int, int], j456: tuple[int, int, int]) -> int:
    """Return the number of values of a bra/ket pair of triples with the given doubled j values."""
    j1, j2, j3 = j123
    j4, j5, j6 = j456
    n = 0
    for j12 in range(abs(j1 - j2) // 2, (j1 + j2) // 2 + 1):
        for j45 in range(abs(j4 - j5) // 2, (j4 + j5) // 2 + 1):
            lo = max(abs(2 * j12 - j3), abs(2 * j45 - j6))
            hi = min(2 * j12 + j3, 2 * j45 + j6)
            if hi >= lo:
                n += (hi - lo) // 2 + 1
    return ISO_BLOCK * n


def tuple_entries(
    j123: tuple[int, int, int], j456: tuple[int, int, int]
) -> Iterator[tuple[int, int, int, int, int, int]]:
    """Yield ``(j12, t12, j45, t45, 2J, 2T)`` of every value of a tuple, in file order."""
    j1, j2, j3 = j123
    j4, j5, j6 = j456
    for j12 in range(abs(j1 - j2) // 2, (j1 + j2) // 2 + 1):
        for j45 in range(abs(j4 - j5) // 2, (j4 + j5) // 2 + 1):
            lo = max(abs(2 * j12 - j3), abs(2 * j45 - j6))
            hi = min(2 * j12 + j3, 2 * j45 + j6)
            for jj in range(lo, hi + 1, 2):
                for t12 in (0, 1):
                    for t45 in (0, 1):
                        for tt in range(1, min(2 * t12 + 1, 2 * t45 + 1) + 1, 2):
                            yield j12, t12, j45, t45, jj, tt


class Layout:
    """Tuple layout of a file with the given limits, restricted to an orbit prefix.

    Parameters
    ----------
    emax_orbits : int
        Use only orbits with ``e <= emax_orbits``. The values of a larger file for such bra triples
        form a prefix of the file (the writer's outermost loop is over ``a``).
    e2max, e3max : int
        Limits of the file.
    """

    def __init__(self, emax_orbits: int, e2max: int, e3max: int) -> None:
        if e3max > e2max:
            raise ValueError("readers and writers need e3max <= e2max (docs/spec.md 1.2)")
        self.orbits = orbit_list(emax_orbits)
        self.triples = triples(self.orbits, e2max, e3max)
        self.jkeys = [
            (self.orbits[t[0]].j2, self.orbits[t[1]].j2, self.orbits[t[2]].j2) for t in self.triples
        ]
        self.parity = np.array([sum(self.orbits[x].l for x in t) % 2 for t in self.triples])
        classes = sorted(set(self.jkeys))
        cid = {k: i for i, k in enumerate(classes)}
        self.cls = np.array([cid[k] for k in self.jkeys])
        n = len(classes)
        self.table = np.zeros((n, n), dtype=np.int64)
        for i, ki in enumerate(classes):
            for k, kk in enumerate(classes):
                self.table[i, k] = pair_count(ki, kk)

    def row(self, i: int) -> np.ndarray:
        """Return the value counts of the kets ``k <= i`` of bra triple ``i``.

        The count is 0 where the parity of the ket differs from that of the bra.
        """
        c = self.table[self.cls[i], self.cls[: i + 1]]
        return np.where(self.parity[: i + 1] == self.parity[i], c, 0)

    def total(self) -> int:
        """Return the number of values of the file prefix covered by the orbit list."""
        return int(sum(self.row(i).sum() for i in range(len(self.triples))))


def count_values(emax: int, e2max: int, e3max: int) -> int:
    """Return the number of values of a file written with ``(emax, e2max, e3max)``."""
    return Layout(emax, e2max, e3max).total()


def kept_intervals(layout: Layout, keep: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return ``(offset, count, bra, ket)`` for every tuple with both triples kept, in file order.

    Parameters
    ----------
    layout : Layout
        Layout of the source file.
    keep : numpy.ndarray
        Boolean per triple.
    """
    out = []
    base = 0
    for i in range(len(layout.triples)):
        c = layout.row(i)
        off = base + np.cumsum(c) - c
        if keep[i]:
            for k in np.nonzero(keep[: i + 1] & (c > 0))[0]:
                out.append((int(off[k]), int(c[k]), i, int(k)))
        base += int(c.sum())
    return out


def keep_mask(layout: Layout, emax: int, e2max: int, e3max: int) -> np.ndarray:
    """Return a boolean per triple of ``layout``: it satisfies ``(emax, e2max, e3max)``."""
    orb = layout.orbits
    return np.array(
        [
            all(orb[x].e <= emax for x in t)
            and max(orb[t[0]].e + orb[t[1]].e, orb[t[0]].e + orb[t[2]].e, orb[t[1]].e + orb[t[2]].e)
            <= e2max
            and sum(orb[x].e for x in t) <= e3max
            for t in layout.triples
        ]
    )


@cache
def _forced_zero_mask(
    j123: tuple[int, int, int],
    j456: tuple[int, int, int],
    ab: bool,
    de: bool,
    abc: bool,
    def_: bool,
) -> np.ndarray:
    """Return a boolean per value of a tuple: the entry is forced to zero (docs/spec.md 1.2).

    The flags tell which orbits coincide: ``ab`` a == b, ``de`` d == e, ``abc`` a == b == c,
    ``def_`` d == e == f.
    """
    ent = np.array(list(tuple_entries(j123, j456)), dtype=np.int64).reshape(-1, 6)
    zero = np.zeros(len(ent), dtype=bool)
    if ab:
        zero |= (ent[:, 0] + ent[:, 1]) % 2 == 0
    if de:
        zero |= (ent[:, 2] + ent[:, 3]) % 2 == 0
    if abc and j123[2] < 3:
        zero |= ent[:, 5] == 3
    if def_ and j456[2] < 3:
        zero |= ent[:, 5] == 3
    zero.setflags(write=False)
    return zero


def _forbidden_ok(layout: Layout, bra: int, ket: int, vals: np.ndarray) -> bool:
    """Return True if the entries forced to zero by the Pauli-type conditions are zero."""
    a, b, c = layout.triples[bra]
    d, e, f = layout.triples[ket]
    mask = _forced_zero_mask(
        layout.jkeys[bra], layout.jkeys[ket], a == b, d == e, a == b == c, d == e == f
    )
    return not np.any(vals[mask] != 0.0)


@dataclass(frozen=True)
class Me3jTuple:
    """One pair of orbit triples of the file with all its values.

    Attributes
    ----------
    bra, ket : tuple[int, int, int]
        0-based orbit indices ``(a, b, c)`` with ``a >= b >= c``; the bra is lexicographically
        not below the ket.
    values : numpy.ndarray
        Shape (n,), MeV, in the order of ``tuple_entries(j123, j456)``.
    """

    bra: tuple[int, int, int]
    ket: tuple[int, int, int]
    values: np.ndarray


def read_values3(path: str | Path) -> np.ndarray:
    """Read all values of a text or gzip ``me3j`` file after the one-line header.

    The array is cached per process (keyed on path, size and modification time) and read-only, so
    the chunks of a sweep handled by one worker parse the file once.

    Parameters
    ----------
    path : str or Path
        The file; gzip is detected from the magic bytes. The whole file is held in memory, so this
        is for the e3max <= 8 files (5.2e6 values at emax 8), not for the large ones.

    Returns
    -------
    numpy.ndarray
        Flat float64 array.
    """
    path = Path(path).resolve()
    st = path.stat()
    return _read_values3(str(path), st.st_size, st.st_mtime_ns)


@cache
def _read_values3(path: str, size: int, mtime_ns: int) -> np.ndarray:
    del size, mtime_ns  # cache key only
    with open(path, "rb") as fh:
        magic = fh.read(2)
    opener = _gz.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as fh:
        fh.readline()
        raw = fh.read()
    out = _fields(raw).copy().view(f"S{FIELD}").ravel().astype(np.float64)
    out.setflags(write=False)
    return out


@cache
def row_plan(emax: int, e2max: int, e3max: int) -> tuple[Layout, np.ndarray]:
    """Return the layout of a file and the value offset of the start of every bra row.

    The file lists the tuples row by row: bra triple ``i`` with all its kets ``k <= i``. The array
    has ``n_triples + 1`` entries; the last is the number of values of the file.
    """
    lay = Layout(emax, e2max, e3max)
    base = np.zeros(len(lay.triples) + 1, dtype=np.int64)
    for i in range(len(lay.triples)):
        base[i + 1] = base[i] + lay.row(i).sum()
    base.setflags(write=False)
    return lay, base


def _parse(raw: bytes) -> np.ndarray:
    """Parse raw ``10f16.8`` text lines into a float64 array."""
    return _fields(raw).copy().view(f"S{FIELD}").ravel().astype(np.float64)


class ValueReader:
    """Reads ranges of the values of a text or gzip ``me3j`` file without loading it all.

    Ranges are requested by value index (position in the file after the header line). A request
    that continues the previous one is served from a buffer; one that jumps ahead or back seeks,
    which in a gzip stream means decompressing from the start (or forward) without parsing. At most
    one block of ``block_lines`` lines beyond the requested range is held.

    Parameters
    ----------
    path : str or Path
        The file; gzip is detected from the magic bytes.
    block_lines : int
        Lines read at a time (default 100,000, about 16 MB of text).
    """

    def __init__(self, path: str | Path, block_lines: int = 100_000) -> None:
        self.path = Path(path)
        with self.path.open("rb") as fh:
            magic = fh.read(2)
        opener = _gz.open if magic == b"\x1f\x8b" else open
        self._opener = opener
        self._open()
        self._block = block_lines
        self._buf = np.empty(0)
        self._b0 = 0
        self._valid = True  # the file position follows the buffer

    def _open(self) -> None:
        self._f = self._opener(self.path, "rb")
        self._hlen = len(self._f.readline())

    def close(self) -> None:
        """Close the file."""
        self._f.close()

    def __enter__(self) -> "ValueReader":
        """Return the reader."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the file."""
        self.close()

    def _seek_line(self, line: int) -> None:
        target = self._hlen + line * LINE_BYTES
        if target < self._f.tell():  # the gzip stream only moves forward: start again
            self._f.close()
            self._open()
        self._f.seek(target)
        self._buf = np.empty(0)
        self._b0 = line * PER_LINE
        self._valid = True

    def get(self, v0: int, v1: int) -> np.ndarray:
        """Return the values with indices ``v0 <= v < v1`` (a view valid until the next call).

        Raises
        ------
        EOFError
            If the file ends before ``v1``.
        """
        end = self._b0 + len(self._buf)
        if not self._valid or v0 < self._b0 or v0 >= end:
            self._seek_line(v0 // PER_LINE)
        elif v0 > self._b0:
            self._buf = self._buf[v0 - self._b0 :]
            self._b0 = v0
        pieces = [self._buf]
        have = len(self._buf)
        while self._b0 + have < v1:
            lines = max(self._block, -(-(v1 - self._b0 - have) // PER_LINE))
            raw = self._f.read(lines * LINE_BYTES)
            if not raw:
                raise EOFError(f"{self.path}: ends before value {v1}")
            vals = _parse(raw)
            pieces.append(vals)
            have += len(vals)
        if len(pieces) > 1:
            self._buf = np.concatenate(pieces)
        return self._buf[v0 - self._b0 : v1 - self._b0]

    def check_end(self, total: int) -> None:
        """Check that the file holds exactly ``total`` values (reads to the end of the stream).

        Raises
        ------
        ValueError
            If there are more or fewer values.
        """
        self._seek_line(total // PER_LINE)
        self._valid = False
        rest = self._f.read()
        tail = total % PER_LINE
        if len(rest.rstrip(b"\n")) != tail * FIELD or rest.count(b"\n") > 1:
            raise ValueError(f"{self.path}: more values than the {total} expected for the limits")


def file_value_count(path: str | Path) -> int:
    """Return the number of values of a ``me3j`` file by reading the stream once.

    Decompression only (about 800 MB/s with ISA-L); the lines are assumed to be ``10f16.8``.

    Raises
    ------
    ValueError
        If the length is not whole lines of ten 16-character fields plus a final partial line.
    """
    path = Path(path)
    with path.open("rb") as fh:
        magic = fh.read(2)
    opener = _gz.open if magic == b"\x1f\x8b" else open
    size = 0
    with opener(path, "rb") as f:
        f.readline()
        while chunk := f.read(1 << 26):
            size += len(chunk)
    full, rem = divmod(size, LINE_BYTES)
    if rem and not ((rem - 1) % FIELD == 0 or rem % FIELD == 0):
        raise ValueError(
            f"{path}: {rem} bytes after the last full line is not a whole number of fields"
        )
    return full * PER_LINE + (rem - (rem % FIELD == 1)) // FIELD if rem else full * PER_LINE


def iter_tuples3(
    path: str | Path,
    emax: int,
    e2max: int,
    e3max: int,
    use_limits: tuple[int, int, int] | None = None,
    row_range: tuple[int, int] | None = None,
    reader: ValueReader | None = None,
) -> Iterator[Me3jTuple]:
    """Yield the orbit-triple pairs of a file with their values, reading the file as a stream.

    Only the values of the requested rows are read (the rest of a gzip stream is skipped by
    decompressing). The total value count is checked when the range reaches the last row.

    Parameters
    ----------
    path : str or Path
        The file.
    emax, e2max, e3max : int
        Limits the file was written with (``e3max <= e2max``).
    use_limits : tuple[int, int, int], optional
        ``(emax, e2max, e3max)`` filter applied on the fly (inclusive), not above the file's.
    row_range : tuple[int, int], optional
        Half-open range of bra rows (positions in the triple list of ``Layout``), to split a file
        into fixed chunks.
    reader : ValueReader, optional
        An open reader to continue from (for consecutive chunks in one process); otherwise the file
        is opened here.

    Yields
    ------
    Me3jTuple
        One pair of triples. The values of a tuple are checked to be zero at the forced-zero
        entries.

    Raises
    ------
    ValueError
        If the file holds a different number of values than the loop implies, or the filter
        exceeds the file's limits.
    RuntimeError
        If a value at a forced-zero entry is nonzero.
    """
    use = (emax, e2max, e3max) if use_limits is None else use_limits
    if use[0] > emax or use[1] > e2max or use[2] > e3max:
        raise ValueError(f"filter {use} exceeds the file limits {(emax, e2max, e3max)}")
    lay, base = row_plan(emax, e2max, e3max)
    keep = keep_mask(lay, *use)
    n = len(lay.triples)
    lo, hi = (0, n) if row_range is None else row_range
    rd = ValueReader(path) if reader is None else reader
    try:
        for i in range(lo, hi):
            if not keep[i]:
                continue
            counts = lay.row(i)
            kets = np.flatnonzero(keep[: i + 1] & (counts > 0))
            offs = base[i] + np.cumsum(counts) - counts
            for k in kets:
                try:
                    vals = rd.get(int(offs[k]), int(offs[k] + counts[k]))
                except EOFError:
                    raise ValueError(
                        f"{path}: fewer values, expected {int(base[n])} for the limits"
                    ) from None
                if not _forbidden_ok(lay, i, int(k), vals):
                    raise RuntimeError(f"nonzero at a forced-zero entry, tuple ({i}, {k})")
                yield Me3jTuple(lay.triples[i], lay.triples[int(k)], vals)
        if hi == n:
            try:
                rd.check_end(int(base[n]))
            except ValueError:
                raise ValueError(
                    f"{path}: extra values, expected {int(base[n])} for the limits"
                ) from None
    finally:
        if reader is None:
            rd.close()


def filter_me3j(
    src: Path,
    dst: Path,
    file_limits: tuple[int, int],
    use_limits: tuple[int, int, int],
    check: bool = True,
) -> dict:
    """Cut a smaller space out of a ``me3j.gz`` file.

    Parameters
    ----------
    src, dst : Path
        Source file (``.gz`` or plain text) and destination (``.gz`` or plain text by suffix).
    file_limits : tuple[int, int]
        ``(e2max, e3max)`` the source was written with. Its emax need not be given: only the orbits
        with ``e <= emax_use`` are enumerated.
    use_limits : tuple[int, int, int]
        ``(emax, e2max, e3max)`` of the output. Must not exceed the source limits.
    check : bool
        Verify that the values at the entries forced to zero are zero in every copied tuple.

    Returns
    -------
    dict
        Counts: values of the source prefix, values written, tuples written.
    """
    e2f, e3f = file_limits
    emax, e2max, e3max = use_limits
    if e2max > e2f or e3max > e3f:
        raise ValueError("requested limits exceed the file's")
    lay = Layout(emax, e2f, e3f)
    keep = keep_mask(lay, emax, e2max, e3max)
    ivs = kept_intervals(lay, keep)
    opener = _gz.open if str(src).endswith(".gz") else open
    wopen = gzip.open if str(dst).endswith(".gz") else open
    # group intervals that share or touch lines, so every line is read once
    groups: list[list[tuple[int, int, int, int]]] = []
    for iv in ivs:
        line0 = iv[0] // PER_LINE
        if groups and line0 <= (groups[-1][-1][0] + groups[-1][-1][1] - 1) // PER_LINE + 1:
            groups[-1].append(iv)
        else:
            groups.append([iv])
    written = 0
    with opener(src, "rb") as f, wopen(dst, "wb") as g:
        header = f.readline()
        g.write(header)
        hlen = len(header)
        parts: list[np.ndarray] = []
        nparts = 0
        for grp in groups:
            first = grp[0][0]
            last = grp[-1][0] + grp[-1][1] - 1
            line0, line1 = first // PER_LINE, last // PER_LINE
            f.seek(hlen + line0 * LINE_BYTES)
            raw = f.read((line1 - line0 + 1) * LINE_BYTES)
            fields = _fields(raw)
            for off, cnt, bra, ket in grp:
                lo = off - line0 * PER_LINE
                sel = fields[lo : lo + cnt]
                if len(sel) != cnt:
                    raise RuntimeError(f"short read at tuple ({bra}, {ket}): {len(sel)} of {cnt}")
                if check:
                    vals = sel.copy().view(f"S{FIELD}").ravel().astype(float)
                    if not _forbidden_ok(lay, bra, ket, vals):
                        raise RuntimeError(f"nonzero at a forced-zero entry, tuple ({bra}, {ket})")
                parts.append(sel)
                nparts += cnt
            if nparts >= 10_000_000:
                allf = np.concatenate(parts)
                written += _emit(g, allf, final=False)
                parts = [allf[(len(allf) // PER_LINE) * PER_LINE :]]
                nparts = len(parts[0])
        if parts:
            written += _emit(g, np.concatenate(parts), final=True)
    return {
        "source_prefix_values": lay.total(),
        "values_written": written,
        "tuples_written": len(ivs),
    }


def _fields(raw: bytes) -> np.ndarray:
    """Split raw lines of ``10f16.8`` text into an array of 16-byte fields, shape (n, 16)."""
    buf = np.frombuffer(raw, dtype=np.uint8)
    lines = len(raw) // LINE_BYTES
    body = buf[: lines * LINE_BYTES].reshape(lines, LINE_BYTES)[:, : FIELD * PER_LINE]
    fields = body.reshape(-1, FIELD)
    tail = raw[lines * LINE_BYTES :].rstrip(b"\n")  # last partial line of the file
    if tail:
        fields = np.vstack([fields, np.frombuffer(tail, dtype=np.uint8).reshape(-1, FIELD)])
    return fields


class _Writer(Protocol):
    def write(self, data: bytes, /) -> object: ...


def _emit(g: _Writer, fields: np.ndarray, final: bool) -> int:
    """Write complete lines of ten fields, and the last partial line if ``final``.

    Returns the number of values written.
    """
    n = len(fields) if final else (len(fields) // PER_LINE) * PER_LINE
    done = 0
    while done < n:
        take = min(PER_LINE * 100_000, n - done)
        chunk = fields[done : done + take]
        nl = -(-take // PER_LINE)
        out = np.full((nl, LINE_BYTES), ord(" "), dtype=np.uint8)
        padded = np.full((nl * PER_LINE, FIELD), ord(" "), dtype=np.uint8)
        padded[:take] = chunk
        out[:, : FIELD * PER_LINE] = padded.reshape(nl, FIELD * PER_LINE)
        out[:, -1] = ord("\n")
        data = out.tobytes()
        if take % PER_LINE:  # partial last line: only its fields, then the newline
            keep = (nl - 1) * LINE_BYTES + (take % PER_LINE) * FIELD
            data = data[:keep] + b"\n"
        g.write(data)
        done += take
    return n


def _git(*args: str) -> str:
    here = Path(__file__).resolve().parent
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=here, check=False
    ).stdout.strip()


def main(argv: list[str] | None = None) -> None:
    """Command line entry point: ``python -m nuham_mscheme.me3j SRC DST --file-e2max ... ``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--file-e2max", type=int, required=True)
    ap.add_argument("--file-e3max", type=int, required=True)
    ap.add_argument("--emax", type=int, required=True)
    ap.add_argument("--e2max", type=int, required=True)
    ap.add_argument("--e3max", type=int, required=True)
    ap.add_argument("--sha256", action="store_true", help="hash the source file (reads it fully)")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)
    stats = filter_me3j(
        args.src,
        args.dst,
        (args.file_e2max, args.file_e3max),
        (args.emax, args.e2max, args.e3max),
        check=not args.no_check,
    )
    expected = count_values(args.emax, args.e2max, args.e3max)
    stats["expected_values"] = expected
    stats["count_ok"] = stats["values_written"] == expected
    manifest = {
        "date": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "source": {
            "path": str(args.src),
            "size": args.src.stat().st_size,
            "mtime": args.src.stat().st_mtime,
            "file_e2max": args.file_e2max,
            "file_e3max": args.file_e3max,
        },
        "output": {
            "path": str(args.dst),
            "emax": args.emax,
            "e2max": args.e2max,
            "e3max": args.e3max,
        },
        "stats": stats,
        "forced_zero_check": not args.no_check,
    }
    if args.sha256:
        h = hashlib.sha256()
        with open(args.src, "rb") as f:
            while chunk := f.read(1 << 26):
                h.update(chunk)
        manifest["source"]["sha256"] = h.hexdigest()
    manifest["output"]["sha256"] = hashlib.sha256(args.dst.read_bytes()).hexdigest()
    Path(str(args.dst) + ".json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(stats))
    if not stats["count_ok"]:
        sys.exit("value count differs from the analytic count")


if __name__ == "__main__":
    main()
