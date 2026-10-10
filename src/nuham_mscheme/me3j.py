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


def _forbidden_ok(layout: Layout, bra: int, ket: int, vals: np.ndarray) -> bool:
    """Return True if the entries forced to zero by the Pauli-type conditions are zero."""
    a, b, c = layout.triples[bra]
    d, e, f = layout.triples[ket]
    j3, j6 = layout.jkeys[bra][2], layout.jkeys[ket][2]
    for v, (j12, t12, j45, t45, _jj, tt) in zip(
        vals, tuple_entries(layout.jkeys[bra], layout.jkeys[ket]), strict=True
    ):
        zero = (
            (a == b and (j12 + t12) % 2 == 0)
            or (d == e and (j45 + t45) % 2 == 0)
            or (a == b == c and tt == 3 and j3 < 3)
            or (d == e == f and tt == 3 and j6 < 3)
        )
        if zero and v != 0.0:
            return False
    return True


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
    keep = np.array(
        [
            all(lay.orbits[x].e <= emax for x in t)
            and max(
                lay.orbits[t[0]].e + lay.orbits[t[1]].e,
                lay.orbits[t[0]].e + lay.orbits[t[2]].e,
                lay.orbits[t[1]].e + lay.orbits[t[2]].e,
            )
            <= e2max
            and sum(lay.orbits[x].e for x in t) <= e3max
            for t in lay.triples
        ]
    )
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
