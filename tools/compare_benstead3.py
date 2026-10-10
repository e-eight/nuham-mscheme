"""Compare a run3 manifest with Benstead et al. (arXiv 2607.21563) Tables IX-XII, per M3 block.

Usage: python tools/compare_benstead3.py runs/out/3n/e8_hw30to16.json

Benstead's columns are "sum |W|" and "nonzero upper-diagonal elements" of one sign of M3;
``docs/data/benstead_tables_ix_xii.json`` holds the values transcribed from the PDF, keyed by 2*M3.
For each block the script prints our counts and 1-norms (upper triangle, tau = 0) and the ratios,
and the smallest threshold tau of the ladder at which our nonzero count falls to theirs (``-`` if
none), with our 1-norm of the elements above that threshold.
"""

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "docs" / "data" / "benstead_tables_ix_xii.json"


def main(manifest: str) -> None:
    """Print the per-M3 comparison for the manifest at ``manifest``."""
    bench = json.loads(DATA.read_text())
    blocks = json.loads(Path(manifest).read_text())["parts"]["w"]["blocks"]
    for sector in ("ppp", "ppn", "pnn", "nnn"):
        print(
            f"{sector}: 2M3  nnz(B)  nnz(mine)  ratio |  lam(B)  lam(mine)  ratio | tau, lam above"
        )
        for m, (lam_b, nnz_b) in sorted(bench[sector].items(), key=lambda kv: int(kv[0])):
            blk = blocks.get(f"{sector}:{m}")
            if blk is None:
                print(f"  {int(m):3d} {nnz_b:9d}   (no such block here)")
                continue
            n, lam = blk["nnz_ut"]["0.0"], blk["lambda_ut"]
            tau = next((t for t, v in blk["nnz_ut"].items() if v <= nnz_b), None)
            cut = "-" if tau is None else f"{tau} {blk['lambda_ut_above'][tau]:.2f}"
            print(
                f"  {int(m):3d} {nnz_b:9d} {n:10d} {n / nnz_b:8.1f} |"
                f" {lam_b:9.2f} {lam:10.2f} {lam / max(lam_b, 1e-9):8.1f} | {cut}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
