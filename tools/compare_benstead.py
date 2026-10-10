"""Compare a run2 manifest with Benstead et al. (arXiv 2607.21563) Tables VI-VIII, per M2 block.

Usage: python tools/compare_benstead.py runs/out/emax8_hw16.json

Benstead's columns are "sum |V|" and "nonzero upper-diagonal elements"; ``docs/data`` holds the
values transcribed from the PDF (M2 = 0 and |M2| >= 1 rows, single-sign rows).
"""

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "docs" / "data" / "benstead_tables_vi_viii.json"


def main(manifest: str) -> None:
    """Print the per-M2 comparison for the manifest at ``manifest``."""
    bench = json.loads(DATA.read_text())
    blocks = json.loads(Path(manifest).read_text())["parts"]["v"]["blocks"]
    for sector in ("pp", "nn", "pn"):
        print(f"{sector}: M2  nnz(B)  nnz(mine)  ratio |  lam(B)  lam(mine)  ratio")
        for m, (lam_b, nnz_b) in sorted(bench[sector].items(), key=lambda kv: int(kv[0])):
            blk = blocks[f"{sector}:{2 * int(m)}"]
            n, lam = blk["nnz_ut"]["0.0"], blk["lambda_ut"]
            print(
                f"  {int(m):3d} {nnz_b:9d} {n:10d} {n / nnz_b:7.1f} |"
                f" {lam_b:10.2f} {lam:10.2f} {lam / lam_b:5.2f}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
