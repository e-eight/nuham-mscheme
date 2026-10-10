"""Compare run2 manifests made from different NN potentials with Benstead et al. Tables VI-VIII.

Usage: python tools/potential_scan.py runs/out/emax8_hw16.json runs/out/potscan/*.json

For each manifest prints: the single PN element at M = 17 (Benstead: 2.09), total lambda and nnz
(upper triangle), and ours/Benstead lambda ratios at selected M for PP and PN. The count does not
depend on the potential (fill is structural), so only lambda is compared per M.
"""

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "docs" / "data" / "benstead_tables_vi_viii.json"
MS = (0, 4, 8, 12, 14, 16)


def main(manifests: list[str]) -> None:
    """Print one comparison row per manifest."""
    bench = json.loads(DATA.read_text())
    head = " ".join(f"pp{m:<3d}" for m in MS) + "  " + " ".join(f"pn{m:<3d}" for m in MS)
    print(f"{'manifest':<22} {'pn M17':>7} {'lam_ut':>9} {'nnz_ut':>9}   {head}")
    for path in manifests:
        man = json.loads(Path(path).read_text())
        bl = man["parts"]["v"]["blocks"]
        tot = man["parts"]["v"]["totals"]["all"]
        ratios = [
            bl[f"{s}:{2 * m}"]["lambda_ut"] / bench[s][str(m)][0] for s in ("pp", "pn") for m in MS
        ]
        cells = " ".join(f"{r:5.2f}" for r in ratios[: len(MS)])
        cells += "  " + " ".join(f"{r:5.2f}" for r in ratios[len(MS) :])
        print(
            f"{Path(path).stem:<22} {bl['pn:34']['lambda_ut']:7.4f} {tot['lambda_ut']:9.3e}"
            f" {tot['nnz_ut']['0.0']:9.3e}   {cells}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
