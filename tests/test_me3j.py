"""Tests of the me3j layout counts and of the offset-based filter."""

import gzip
import zlib
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from nuham_mscheme.me3j import (
    ValueReader,
    count_values,
    file_value_count,
    filter_me3j,
    iter_tuples3,
    read_values3,
    row_plan,
    tuple_entries,
)
from nuham_mscheme.orbits import orbit_list

HEADER = "NNN int. calculated by test\n"
ORBITS = orbit_list(12)


def _loop(emax: int, e2max: int, e3max: int):
    """Port of the writer's nested loops (docs/spec.md 1.2, spec-5d 6.2); yields tuple keys."""
    orb = orbit_list(emax)
    n = len(orb)
    for i1 in range(n):
        for i2 in range(i1 + 1):
            if orb[i1].e + orb[i2].e > e2max:
                continue
            for i3 in range(i2 + 1):
                e1, e2, e3 = orb[i1].e, orb[i2].e, orb[i3].e
                if e1 + e3 > e2max or e2 + e3 > e2max or e1 + e2 + e3 > e3max:
                    continue
                p123 = (orb[i1].l + orb[i2].l + orb[i3].l) % 2
                for i4 in range(i1 + 1):
                    i5max = i2 if i4 == i1 else i4
                    for i5 in range(i5max + 1):
                        if orb[i4].e + orb[i5].e > e2max:
                            continue
                        i6max = i3 if (i4 == i1 and i5 == i2) else i5
                        for i6 in range(i6max + 1):
                            e4, e5, e6 = orb[i4].e, orb[i5].e, orb[i6].e
                            if e4 + e6 > e2max or e5 + e6 > e2max or e4 + e5 + e6 > e3max:
                                continue
                            if p123 != (orb[i4].l + orb[i5].l + orb[i6].l) % 2:
                                continue
                            j1, j2, j3 = orb[i1].j2, orb[i2].j2, orb[i3].j2
                            j4, j5, j6 = orb[i4].j2, orb[i5].j2, orb[i6].j2
                            for j12 in range(abs(j1 - j2) // 2, (j1 + j2) // 2 + 1):
                                for j45 in range(abs(j4 - j5) // 2, (j4 + j5) // 2 + 1):
                                    lo = max(abs(2 * j12 - j3), abs(2 * j45 - j6))
                                    hi = min(2 * j12 + j3, 2 * j45 + j6)
                                    for jj in range(lo, hi + 1, 2):
                                        for t12 in (0, 1):
                                            for t45 in (0, 1):
                                                for tt in range(
                                                    1, min(2 * t12 + 1, 2 * t45 + 1) + 1, 2
                                                ):
                                                    yield (
                                                        (i1, i2, i3, i4, i5, i6),
                                                        (j12, t12, j45, t45, jj, tt),
                                                    )


def _value(key: tuple, entry: tuple) -> float:
    """Deterministic pseudo value of a tuple entry, zero at the forced-zero entries."""
    i1, i2, i3, i4, i5, i6 = key
    j12, t12, j45, t45, jj, tt = entry
    orb = ORBITS
    if i1 == i2 and (j12 + t12) % 2 == 0:
        return 0.0
    if i4 == i5 and (j45 + t45) % 2 == 0:
        return 0.0
    if i1 == i2 == i3 and tt == 3 and orb[i3].j2 < 3:
        return 0.0
    if i4 == i5 == i6 and tt == 3 and orb[i6].j2 < 3:
        return 0.0
    h = zlib.crc32(repr((key, entry)).encode())
    return (h % 2_000_001 - 1_000_000) / 1e5  # |x| <= 10 with five decimals


@cache
def _text(emax: int, e2max: int, e3max: int) -> bytes:
    vals = [_value(k, e) for k, e in _loop(emax, e2max, e3max)]
    lines = ["".join(f"{v:16.8f}" for v in vals[i : i + 10]) for i in range(0, len(vals), 10)]
    return (HEADER + "\n".join(lines) + "\n").encode()


@pytest.mark.parametrize(
    "limits,expected",
    [((4, 8, 4), 24890), ((6, 12, 6), 456320), ((8, 16, 8), 5199790), ((10, 20, 12), 273181225)],
)
def test_value_counts_match_the_spec(limits, expected):
    assert count_values(*limits) == expected


def test_count_matches_the_explicit_loops():
    for limits in [(3, 6, 6), (4, 8, 5), (4, 6, 6)]:
        assert count_values(*limits) == sum(1 for _ in _loop(*limits))


@pytest.mark.parametrize(
    "use",
    [(4, 8, 4), (3, 6, 6), (4, 8, 5), (2, 4, 4), (4, 6, 3)],
)
def test_filter_equals_direct_write(tmp_path: Path, use):
    src, dst = tmp_path / "src.me3j.gz", tmp_path / "dst.me3j"
    with gzip.open(src, "wb") as f:
        f.write(_text(4, 8, 8))
    stats = filter_me3j(src, dst, (8, 8), use)
    assert stats["values_written"] == count_values(*use)
    assert dst.read_bytes() == _text(*use)


def test_filter_of_a_filter_is_the_filter(tmp_path: Path):
    src, mid, a, b = (tmp_path / n for n in ("s.me3j", "m.me3j", "a.me3j", "b.me3j"))
    src.write_bytes(_text(4, 8, 8))
    filter_me3j(src, mid, (8, 8), (4, 8, 6))
    filter_me3j(mid, a, (8, 6), (3, 6, 4))
    filter_me3j(src, b, (8, 8), (3, 6, 4))
    assert a.read_bytes() == b.read_bytes()


def test_misaligned_stream_is_detected(tmp_path: Path):
    """A file whose forced-zero entries are nonzero must be refused."""
    src, dst = tmp_path / "bad.me3j", tmp_path / "out.me3j"
    text = _text(3, 6, 6).decode().replace("0.00000000", "1.00000000")
    src.write_text(text)
    with pytest.raises(RuntimeError, match="forced-zero"):
        filter_me3j(src, dst, (6, 6), (3, 6, 6))


def test_reader_returns_the_values_in_file_order(tmp_path: Path):
    path = tmp_path / "t.me3j"
    path.write_bytes(_text(4, 8, 8))
    expected = np.array([float(x) for x in _text(4, 8, 8).decode().split()[len(HEADER.split()) :]])
    assert np.array_equal(read_values3(path), expected)
    tuples = list(iter_tuples3(path, 4, 8, 8))
    assert np.array_equal(np.concatenate([t.values for t in tuples]), expected)
    for t, ((bra, ket), _) in zip(tuples, _tuple_keys(4, 8, 8), strict=True):
        assert (t.bra, t.ket) == (bra, ket)
    assert all(len(t.values) == len(list(tuple_entries(_j(t.bra), _j(t.ket)))) for t in tuples)


def _j(triple):
    return tuple(ORBITS[i].j2 for i in triple)


def _tuple_keys(emax: int, e2max: int, e3max: int):
    """Distinct (bra, ket) triples in file order, from the independent loop port."""
    last = None
    for key, entry in _loop(emax, e2max, e3max):
        k = (key[:3], key[3:])
        if k != last:
            last = k
            yield k, entry


def test_reader_filter_equals_reading_the_filtered_file(tmp_path: Path):
    big, small = tmp_path / "big.me3j", tmp_path / "small.me3j"
    big.write_bytes(_text(4, 8, 8))
    small.write_bytes(_text(3, 6, 5))
    a = list(iter_tuples3(big, 4, 8, 8, use_limits=(3, 6, 5)))
    b = list(iter_tuples3(small, 3, 6, 5))
    assert [(t.bra, t.ket) for t in a] == [(t.bra, t.ket) for t in b]
    assert all(np.array_equal(x.values, y.values) for x, y in zip(a, b, strict=True))


def test_reader_chunks_partition_the_file(tmp_path: Path):
    path = tmp_path / "t.me3j"
    path.write_bytes(_text(3, 6, 6))
    whole = [(t.bra, t.ket) for t in iter_tuples3(path, 3, 6, 6)]
    n = len(row_plan(3, 6, 6)[0].triples)
    parts = [
        (t.bra, t.ket)
        for lo, hi in [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]
        for t in iter_tuples3(path, 3, 6, 6, row_range=(lo, hi))
    ]
    assert parts == whole


@pytest.mark.parametrize("gz", [False, True])
def test_value_reader_ranges_equal_the_whole_file(tmp_path: Path, gz: bool):
    """Random ranges, forward, backward and across blocks, in text and gzip, with tiny blocks."""
    text = _text(3, 6, 6)
    path = tmp_path / ("t.me3j.gz" if gz else "t.me3j")
    path.write_bytes(gzip.compress(text) if gz else text)
    expected = np.array([float(x) for x in text.decode().split()[len(HEADER.split()) :]])
    rng = np.random.default_rng(5)
    with ValueReader(path, block_lines=7) as rd:
        # sequential requests that continue each other, ending with the partial last line
        pos = 0
        while pos < len(expected):
            n = int(rng.integers(1, 40))
            assert np.array_equal(rd.get(pos, min(pos + n, len(expected))), expected[pos : pos + n])
            pos += n
        # jumps in both directions
        for _ in range(40):
            v0 = int(rng.integers(0, len(expected) - 1))
            v1 = int(rng.integers(v0 + 1, min(v0 + 200, len(expected)) + 1))
            assert np.array_equal(rd.get(v0, v1), expected[v0:v1])
        with pytest.raises(EOFError):
            rd.get(len(expected) - 3, len(expected) + 5)
        rd.check_end(len(expected))
        with pytest.raises(ValueError, match="more values"):
            rd.check_end(len(expected) - 4)


def test_file_value_count(tmp_path: Path):
    for lim in [(3, 6, 6), (4, 8, 4)]:
        path = tmp_path / "t.me3j.gz"
        path.write_bytes(gzip.compress(_text(*lim)))
        assert file_value_count(path) == count_values(*lim)


def test_streaming_reader_matches_the_cached_reader(tmp_path: Path):
    path = tmp_path / "t.me3j"
    path.write_bytes(_text(4, 8, 8))
    whole = read_values3(path)
    pos = 0
    for t in iter_tuples3(path, 4, 8, 8):
        assert np.array_equal(t.values, whole[pos : pos + len(t.values)])
        pos += len(t.values)
    assert pos == len(whole)


def test_reader_rejects_a_wrong_limit_a_wrong_count_and_a_filter_beyond_the_file(tmp_path: Path):
    path = tmp_path / "t.me3j"
    text = _text(3, 6, 6)
    path.write_bytes(text)
    # a wrong limit misaligns the stream: caught by the forced zeros or by the value count
    with pytest.raises((ValueError, RuntimeError)):
        list(iter_tuples3(path, 3, 6, 5))
    with pytest.raises(ValueError, match="exceeds"):
        list(iter_tuples3(path, 3, 6, 6, use_limits=(3, 6, 7)))
    # values appended / missing at the end are caught by the value count
    more, fewer = tmp_path / "more.me3j", tmp_path / "fewer.me3j"
    more.write_bytes(text + b"      0.00000000      0.00000000\n")
    fewer.write_bytes(text[:-33])
    with pytest.raises(ValueError, match="extra values, expected"):
        list(iter_tuples3(more, 3, 6, 6))
    with pytest.raises(ValueError, match="fewer values, expected"):
        list(iter_tuples3(fewer, 3, 6, 6))
