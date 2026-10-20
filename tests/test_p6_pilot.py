"""Verify disjoint-support reconstruction and rejection of missing/wrong evidence."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
from test_mscheme3 import _synthetic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/comparison"))
spec = importlib.util.spec_from_file_location(
    "p6_pilot", Path(__file__).resolve().parents[1] / "tools/comparison/p6_pilot.py"
)
assert spec is not None and spec.loader is not None
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

from nuham_mscheme.me3j import iter_tuples3  # noqa: E402
from nuham_mscheme.mscheme3 import TripleBasis  # noqa: E402
from nuham_mscheme.orbits import orbit_list  # noqa: E402
from nuham_mscheme.run3 import three_body_stats  # noqa: E402
from nuham_mscheme.stats3 import ThreeBodyStats  # noqa: E402


@pytest.mark.parametrize("use", [(2, 4, 4), (1, 2, 3), (0, 0, 0)])
def test_support_partition_and_corruption(tmp_path, use):
    """Independent selection plus the filtered driver reconstructs an unfiltered sweep."""
    limits = (2, 4, 4)
    path = tmp_path / "toy.me3j"
    # Reader pseudo-data are not Hermitian. Use the physical seeded fixture:
    # upper = (full + diagonal) / 2 needs symmetric coefficient support.
    values = [float(v) for tup in _synthetic() for v in tup.values]
    lines = ["".join(f"{v:16.8f}" for v in values[i : i + 10]) for i in range(0, len(values), 10)]
    path.write_text("NNN int. physical synthetic fixture\n" + "\n".join(lines) + "\n")
    full = p.parts(three_body_stats(path, limits, n_chunks=8))
    kept = p.parts(three_body_stats(path, limits, use, n_chunks=8))
    removed, counts = p.discarded_stats(path, limits, use)
    complement = p.parts(removed)
    p.reconstruct(full, kept, complement)
    bad = copy.deepcopy(complement)
    bad["w"]["totals"]["all"]["nnz_ut"]["0.0"] += 1
    with pytest.raises(ValueError, match="complement count"):
        p.reconstruct(full, kept, bad)
    if use == limits:
        assert counts["excluded_tuples"] == 0
    elif use == (1, 2, 3):
        # Wrong selection: discard only when BOTH endpoints are outside, losing
        # all cross-boundary coefficients. Structural reconstruction must fail.
        basis = TripleBasis(orbit_list(limits[0]))
        wrong = ThreeBodyStats(basis)
        for tup in iter_tuples3(path, *limits):
            bra = p.retained(tuple(basis.orbits[i].e for i in tup.bra), use)
            ket = p.retained(tuple(basis.orbits[i].e for i in tup.ket), use)
            if not bra and not ket:
                wrong.add_tuple(tup)
        with pytest.raises(ValueError):
            p.reconstruct(full, kept, p.parts(wrong))


def test_gate_rejects_wrong_or_missing_manifest(tmp_path):
    bad = tmp_path / "manifest.json"
    with pytest.raises(FileNotFoundError):
        p.gate(bad)
    bad.write_text('{"status": "passed"}')
    with pytest.raises(ValueError, match="manifest hash"):
        p.gate(bad)
