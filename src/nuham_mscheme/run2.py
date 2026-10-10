"""Driver: two-body m-scheme statistics of a ``me2j`` file, with a provenance manifest.

The file is split into a fixed number of chunks of consecutive quadruples. Each chunk is reduced
independently and the chunks are merged in order, so the result does not depend on the number of
worker processes (CONVENTIONS.md: no float result may change with thread count).
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
from nuham_mscheme.me2j import iter_tuples, tuple_count
from nuham_mscheme.mscheme2 import SECTORS, TInt
from nuham_mscheme.onebody import one_body_stats
from nuham_mscheme.orbits import orbit_list
from nuham_mscheme.stats2 import PARTS, TAUS, BlockStats, TwoBodyStats

N_CHUNKS = 256


def _chunk_job(args: tuple) -> TwoBodyStats:
    path, emax, e2max, emax_use, e2max_use, tint, lo, hi = args
    orbits = orbit_list(emax)
    stats = TwoBodyStats(tint)
    for tup in iter_tuples(path, emax, e2max, emax_use, e2max_use, index_range=(lo, hi)):
        stats.add_tuple(orbits, tup)
    return stats


def two_body_stats(
    path: str | Path,
    emax: int,
    e2max: int,
    emax_use: int | None = None,
    e2max_use: int | None = None,
    tint: TInt | None = None,
    workers: int = 1,
) -> TwoBodyStats:
    """Reduce a file to its two-body statistics.

    Parameters
    ----------
    path : str or Path
        The ``me2j`` file.
    emax, e2max : int
        Limits the file was written with.
    emax_use, e2max_use : int, optional
        Filters applied on the fly (inclusive).
    tint : TInt, optional
        Also track the two-body T_int part and the sum with V.
    workers : int
        Worker processes; the result does not depend on this.

    Returns
    -------
    TwoBodyStats
        The merged statistics.
    """
    if (emax_use or 0) > emax or (e2max_use or 0) > e2max:
        raise ValueError(
            f"filters (emax_use={emax_use}, e2max_use={e2max_use}) exceed the file limits "
            f"({emax}, {e2max})"
        )
    n = tuple_count(emax, e2max)
    bounds = np.linspace(0, n, N_CHUNKS + 1).astype(int)
    jobs = [
        (str(path), emax, e2max, emax_use, e2max_use, tint, int(lo), int(hi))
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    total = TwoBodyStats(tint)
    if workers <= 1:
        results = map(_chunk_job, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_chunk_job, jobs)
    for part in results:
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
    for sector in (*SECTORS, "all"):
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


def require_clean_tree(allow_dirty: bool) -> None:
    """Raise if the working tree is dirty, so a long run fails at its start, not at the manifest."""
    if _git("status", "--porcelain") and not allow_dirty:
        raise RuntimeError("working tree is dirty; commit first or pass allow_dirty (tests only)")


def build_manifest(
    path: Path,
    emax: int,
    e2max: int,
    emax_use: int | None,
    e2max_use: int | None,
    tint: TInt | None,
    stats: TwoBodyStats,
    allow_dirty: bool,
) -> dict:
    """Return the manifest dictionary (docs/spec.md section 7)."""
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("working tree is dirty; commit first or pass allow_dirty (tests only)")
    use_emax = emax if emax_use is None else emax_use
    manifest: dict = {
        "date": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": dirty,
        "input": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "file_limits": {"emax": emax, "e2max": e2max, "lmax": emax},
        "filters": {"emax": use_emax, "e2max": e2max if e2max_use is None else e2max_use},
        "taus": list(TAUS),
        "convention": "B (docs/spec.md 4.2): unique pair states, lambda over both triangles",
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "tint": None if tint is None else {"hw": tint.hw, "A": tint.a},
        "parts": {},
    }
    for part in PARTS:
        if part not in stats.parts:
            continue
        blocks = stats.blocks(part)
        manifest["parts"][part] = {
            "totals": _totals(blocks),
            "blocks": {f"{s}:{m}": _block_dict(b) for (s, m), b in sorted(blocks.items())},
        }
    if tint is not None:
        ob = one_body_stats(orbit_list(use_emax), tint.hw, tint.a)
        manifest["one_body_tint"] = {
            "totals": {
                "lambda_full": float(np.sum([b.lambda_full for b in ob.values()])),
                "lambda_ut": float(np.sum([b.lambda_ut for b in ob.values()])),
                "nnz_full": {
                    str(t): sum(b.nnz_full[i] for b in ob.values()) for i, t in enumerate(TAUS)
                },
                "nnz_ut": {
                    str(t): sum(b.nnz_ut(i) for b in ob.values()) for i, t in enumerate(TAUS)
                },
            },
            "blocks": {f"{s}:{m}": _block_dict(b) for (s, m), b in sorted(ob.items())},
        }
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--emax", type=int, required=True, help="emax of the file")
    ap.add_argument("--e2max", type=int, required=True, help="e2max of the file")
    ap.add_argument("--emax-use", type=int)
    ap.add_argument("--e2max-use", type=int)
    ap.add_argument(
        "--nmax",
        type=int,
        help="matched Nmax space (docs/spec.md 5): sets emax-use and e2max-use; needs --e-fermi",
    )
    ap.add_argument("--e-fermi", type=int, help="Fermi shell e (0 for 4He, 1 for 16O)")
    ap.add_argument("--hw", type=float, help="hbar*omega in MeV (enables T_int)")
    ap.add_argument("--mass", type=int, help="mass number A (enables T_int)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True, help="manifest .json to write")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    require_clean_tree(args.allow_dirty)
    if args.nmax is not None:
        if args.e_fermi is None or args.emax_use is not None or args.e2max_use is not None:
            ap.error("--nmax needs --e-fermi and excludes --emax-use/--e2max-use")
        args.emax_use, args.e2max_use, _ = matched_cutoffs(args.e_fermi, args.nmax)
    tint = TInt(args.hw, args.mass) if args.hw and args.mass else None
    stats = two_body_stats(
        args.file, args.emax, args.e2max, args.emax_use, args.e2max_use, tint, args.workers
    )
    manifest = build_manifest(
        args.file,
        args.emax,
        args.e2max,
        args.emax_use,
        args.e2max_use,
        tint,
        stats,
        args.allow_dirty,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
