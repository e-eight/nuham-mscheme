"""Driver: three-body m-scheme statistics of a ``me3j`` file (exact sweep), with a manifest.

The file's tuple sequence is split into a fixed number of chunks of equal estimated cost. Each
chunk is reduced independently and the chunks are merged in order, so the result does not depend
on the number of worker processes (CONVENTIONS.md: no float result may change with thread count).
"""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from nuham_mscheme.filters import matched_cutoffs
from nuham_mscheme.me3j import file_plan, iter_tuples3, keep_mask
from nuham_mscheme.mscheme3 import SECTORS3, TripleBasis
from nuham_mscheme.orbits import orbit_list
from nuham_mscheme.stats2 import TAUS, BlockStats
from nuham_mscheme.stats3 import ThreeBodyStats

N_CHUNKS = 256


def chunk_bounds(
    limits: tuple[int, int, int], use: tuple[int, int, int], n_chunks: int = N_CHUNKS
) -> list[int]:
    """Return ``n_chunks + 1`` positions that split the file's tuple sequence into equal-cost parts.

    The cost of a tuple is estimated by its number of product states, ``prod(2j+1)`` of the bra
    times that of the ket, and is zero for tuples removed by the filter. The result depends only on
    the limits, not on the number of workers.

    Parameters
    ----------
    limits : tuple[int, int, int]
        ``(emax, e2max, e3max)`` of the file.
    use : tuple[int, int, int]
        ``(emax, e2max, e3max)`` filter.
    n_chunks : int
        Number of chunks.
    """
    lay, plan = file_plan(*limits)
    keep = keep_mask(lay, *use)
    dims = np.array(
        [np.prod([lay.orbits[i].j2 + 1 for i in t]) for t in lay.triples], dtype=np.float64
    )
    w = np.array([dims[b] * dims[k] if keep[b] and keep[k] else 0.0 for _, _, b, k in plan])
    c = np.cumsum(w)
    inner = np.searchsorted(c, np.linspace(0.0, c[-1], n_chunks + 1)[1:-1], side="left")
    return [0, *np.maximum.accumulate(inner).tolist(), len(plan)]


def _chunk_job(args: tuple) -> ThreeBodyStats:
    path, limits, use, lo, hi = args
    basis = TripleBasis(orbit_list(use[0]))
    stats = ThreeBodyStats(basis)
    for tup in iter_tuples3(path, *limits, use_limits=use, index_range=(lo, hi)):
        stats.add_tuple(tup)
    return stats


def three_body_stats(
    path: str | Path,
    limits: tuple[int, int, int],
    use: tuple[int, int, int] | None = None,
    workers: int = 1,
    n_chunks: int = N_CHUNKS,
) -> ThreeBodyStats:
    """Reduce a file to its three-body statistics.

    Parameters
    ----------
    path : str or Path
        The ``me3j`` file.
    limits : tuple[int, int, int]
        ``(emax, e2max, e3max)`` the file was written with (``e3max <= e2max``).
    use : tuple[int, int, int], optional
        ``(emax, e2max, e3max)`` filter, inclusive; defaults to the file's limits.
    workers : int
        Worker processes; the result does not depend on this.
    n_chunks : int
        Number of chunks (fixes the order of the compensated sums).

    Returns
    -------
    ThreeBodyStats
        The merged statistics.
    """
    use = limits if use is None else use
    if any(u > f for u, f in zip(use, limits, strict=True)):
        raise ValueError(f"filters {use} exceed the file limits {limits}")
    bounds = chunk_bounds(limits, use, n_chunks)
    jobs = [
        (str(path), limits, use, lo, hi)
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        if hi > lo
    ]
    total = ThreeBodyStats(TripleBasis(orbit_list(use[0])))
    if workers <= 1:
        for part in map(_chunk_job, jobs):
            total.merge(part)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(_chunk_job, jobs):
                total.merge(part)
    return total


def _block_dict(st: BlockStats) -> dict:
    return {
        "structural_full": st.structural_full,
        "structural_ut": st.structural_ut(),
        "lambda_full": st.lambda_full,
        "lambda_ut": st.lambda_ut,
        "lambda_diag": st.lambda_diag,
        "nnz_full": {str(t): st.nnz_full[i] for i, t in enumerate(TAUS)},
        "nnz_ut": {str(t): st.nnz_ut(i) for i, t in enumerate(TAUS)},
        "nnz_diag": {str(t): st.nnz_diag[i] for i, t in enumerate(TAUS)},
        "lambda_ut_above": {str(t): st.lambda_above_ut(i) for i, t in enumerate(TAUS)},
        "max_abs": st.vmax,
    }


def _totals(blocks: dict[tuple[str, int], BlockStats]) -> dict:
    out: dict = {}
    for sector in (*SECTORS3, "all"):
        sel = [b for (s, _), b in blocks.items() if sector in (s, "all")]
        out[sector] = {
            "structural_ut": sum(b.structural_ut() for b in sel),
            "lambda_full": float(np.sum([b.lambda_full for b in sel])) if sel else 0.0,
            "lambda_ut": float(np.sum([b.lambda_ut for b in sel])) if sel else 0.0,
            "nnz_full": {str(t): sum(b.nnz_full[i] for b in sel) for i, t in enumerate(TAUS)},
            "nnz_ut": {str(t): sum(b.nnz_ut(i) for b in sel) for i, t in enumerate(TAUS)},
            "lambda_ut_above": {
                str(t): float(np.sum([b.lambda_above_ut(i) for b in sel])) if sel else 0.0
                for i, t in enumerate(TAUS)
            },
            "max_abs": max((b.vmax for b in sel), default=0.0),
        }
    return out


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    path: Path,
    limits: tuple[int, int, int],
    use: tuple[int, int, int],
    stats: ThreeBodyStats,
    allow_dirty: bool,
) -> dict:
    """Return the manifest dictionary (docs/spec.md section 7)."""
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("working tree is dirty; commit first or pass allow_dirty (tests only)")
    blocks = stats.blocks()
    return {
        "date": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "input": {"path": str(path), "sha256": _sha256(path)},
        "file_limits": dict(zip(("emax", "e2max", "e3max"), limits, strict=True)),
        "filters": dict(zip(("emax", "e2max", "e3max"), use, strict=True)),
        "taus": list(TAUS),
        "convention": (
            "B (docs/spec.md 4.2): unique triple states p<q<r, lambda over both triangles; "
            "structural = slots inside file tuples (equal parity, common J)"
        ),
        "n_chunks": N_CHUNKS,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "parts": {
            "w": {
                "totals": _totals(blocks),
                "blocks": {f"{s}:{m}": _block_dict(b) for (s, m), b in sorted(blocks.items())},
            }
        },
    }


def main(argv: list[str] | None = None) -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--emax", type=int, required=True, help="emax of the file")
    ap.add_argument("--e2max", type=int, required=True, help="e2max of the file")
    ap.add_argument("--e3max", type=int, required=True, help="e3max of the file")
    ap.add_argument("--emax-use", type=int)
    ap.add_argument("--e2max-use", type=int)
    ap.add_argument("--e3max-use", type=int)
    ap.add_argument(
        "--nmax",
        type=int,
        help="matched Nmax space (docs/spec.md 5): sets the three filters; needs --e-fermi",
    )
    ap.add_argument("--e-fermi", type=int, help="Fermi shell e (0 for 4He, 1 for 16O)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True, help="manifest .json to write")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    limits = (args.emax, args.e2max, args.e3max)
    use = (
        args.emax if args.emax_use is None else args.emax_use,
        args.e2max if args.e2max_use is None else args.e2max_use,
        args.e3max if args.e3max_use is None else args.e3max_use,
    )
    if args.nmax is not None:
        if args.e_fermi is None or any(
            x is not None for x in (args.emax_use, args.e2max_use, args.e3max_use)
        ):
            ap.error("--nmax needs --e-fermi and excludes the --*-use options")
        use = matched_cutoffs(args.e_fermi, args.nmax)
    stats = three_body_stats(args.file, limits, use, args.workers)
    manifest = build_manifest(args.file, limits, use, stats, args.allow_dirty)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
