"""Test two candidate explanations of Benstead et al.'s Tables VI-VIII against our emax 8 file.

Usage: python tools/benstead_hypotheses.py runs/inputs/N3LO_EM500_srg2_hw16_emax8_e2max16.me2j.gz

Test 1 (incomplete decoupling): their per-M counts are J-coupled elements, not m-scheme ones.
  Per pair M, count the file's nonzero coupled elements (one per file quadruple and J, so the
  upper triangle with the diagonal) with J == M ("1a") and with J >= M ("1b"), and sum their |V|.
  pp uses the pp column; pn uses the T=0 and pn T=1 columns, each counted as one element.
Test 2 (species-label collision in pn): their pn elements lose which orbit carries the proton.
  "2a" keeps only the (p, n) assignment of each orbit-sorted pair; "2b" sums the blocks of all
  species assignments onto that one key. Both use the package's m-scheme expansion and
  upper-triangle statistics.

Prints ours/Benstead ratios of nnz (tau = 0) and lambda (upper triangle) per M. A hypothesis
fits if both ratios are about 1 in every block.
"""

import json
import sys
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from nuham_mscheme.me2j import iter_tuples, tuple_count
from nuham_mscheme.mscheme2 import PN, QuadBlock, quad_blocks
from nuham_mscheme.orbits import orbit_list
from nuham_mscheme.stats2 import TwoBodyStats

DATA = Path(__file__).resolve().parents[1] / "docs" / "data" / "benstead_tables_vi_viii.json"
EMAX, E2MAX = 8, 16
MMAX = 20


def _coupled(path: str, lo: int, hi: int) -> dict[str, np.ndarray]:
    """Return per-M coupled counts and 1-norms for J == M and J >= M, pp and pn, in [lo, hi)."""
    out = {
        k: np.zeros(MMAX + 1)
        for k in (
            "pp_eq_n",
            "pp_eq_l",
            "pp_ge_n",
            "pp_ge_l",
            "pn_eq_n",
            "pn_eq_l",
            "pn_ge_n",
            "pn_ge_l",
        )
    }
    for tup in iter_tuples(path, EMAX, E2MAX, index_range=(lo, hi)):
        for k, row in enumerate(tup.values):
            j = tup.jmin + k
            for sec, vals in (("pp", (row[3],)), ("pn", (row[0], row[2]))):
                for v in vals:
                    if v == 0.0:
                        continue
                    out[f"{sec}_eq_n"][j] += 1
                    out[f"{sec}_eq_l"][j] += abs(v)
                    out[f"{sec}_ge_n"][: j + 1] += 1
                    out[f"{sec}_ge_l"][: j + 1] += abs(v)
    return out


def _collision(path: str, lo: int, hi: int) -> tuple[TwoBodyStats, TwoBodyStats]:
    """Return pn statistics for variants 2a (one assignment) and 2b (assignments summed)."""
    orbits = orbit_list(EMAX)
    st_a, st_b = TwoBodyStats(), TwoBodyStats()
    for tup in iter_tuples(path, EMAX, E2MAX, index_range=(lo, hi)):
        diag = tup.a == tup.c and tup.b == tup.d
        pn = [b for b in quad_blocks(orbits, tup) if b.sector == PN]
        first = next(b for b in pn if b.bra_assign == 0 and b.ket_assign == 0)
        st_a._add_block("v", first, first.x, diag)
        summed = QuadBlock(
            PN, 0, 0, sum(b.x for b in pn), first.mb2, first.mk2, first.valid_b, first.valid_k
        )
        st_b._add_block("v", summed, summed.x, diag)
    return st_a, st_b


def _job(args: tuple[str, int, int]) -> tuple:
    path, lo, hi = args
    return _coupled(path, lo, hi), _collision(path, lo, hi)


def main(path: str, workers: int = 8) -> None:
    """Run both tests on the emax 8 file at ``path`` and print the per-M comparison."""
    n = tuple_count(EMAX, E2MAX)
    edges = np.linspace(0, n, 4 * workers + 1).astype(int)
    with Pool(workers) as pool:
        parts = pool.map(
            _job, [(path, int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
        )
    coup = {k: sum(p[0][k] for p in parts) for k in parts[0][0]}
    st_a, st_b = parts[0][1]
    for _, (a, b) in parts[1:]:
        st_a.merge(a)
        st_b.merge(b)
    bl_a, bl_b = st_a.blocks(), st_b.blocks()
    bench = json.loads(DATA.read_text())

    def show(title: str, sector: str, get: Callable[[int], tuple[float, float]]) -> None:
        print(f"\n{title}\n  M  nnz(B)  nnz(test)  ratio |  lam(B)  lam(test)  ratio")
        for m, (lam_b, nnz_b) in sorted(bench[sector].items(), key=lambda kv: int(kv[0])):
            nn, lam = get(int(m))
            print(
                f"{int(m):3d} {nnz_b:9d} {int(nn):10d} {nn / nnz_b:7.2f} |"
                f" {lam_b:10.2f} {lam:10.2f} {lam / lam_b:6.2f}"
            )

    for sec in ("pp", "pn"):
        show(
            f"Test 1a, {sec}: coupled elements with J == M",
            sec,
            lambda m, s=sec: (coup[f"{s}_eq_n"][m], coup[f"{s}_eq_l"][m]),
        )
        show(
            f"Test 1b, {sec}: coupled elements with J >= M",
            sec,
            lambda m, s=sec: (coup[f"{s}_ge_n"][m], coup[f"{s}_ge_l"][m]),
        )
    for title, bl in (
        ("Test 2a, pn: (p, n) assignment only", bl_a),
        ("Test 2b, pn: species assignments summed onto one key", bl_b),
    ):
        show(title, "pn", lambda m, b=bl: (b[("pn", 2 * m)].nnz_ut(0), b[("pn", 2 * m)].lambda_ut))


if __name__ == "__main__":
    main(sys.argv[1])
