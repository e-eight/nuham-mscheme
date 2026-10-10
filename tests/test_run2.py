import json

import numpy as np
import pytest

from nuham_mscheme.mscheme2 import TInt
from nuham_mscheme.run2 import build_manifest, two_body_stats


def test_workers_do_not_change_result(e4_file):
    tint = TInt(16.0, 16)
    a = two_body_stats(e4_file, 4, 8, tint=tint, workers=1)
    b = two_body_stats(e4_file, 4, 8, tint=tint, workers=3)
    for part in a.parts:
        ba, bb = a.blocks(part), b.blocks(part)
        assert ba.keys() == bb.keys()
        for k in ba:
            assert ba[k] == bb[k]  # bitwise, including the 1-norm sums


def test_time_reversal_symmetry(e4_file):
    blocks = two_body_stats(e4_file, 4, 8).blocks("v")
    for (sector, m2), b in blocks.items():
        other = blocks[(sector, -m2)]
        assert b.lambda_full == pytest.approx(other.lambda_full, rel=1e-12)
        assert b.nnz_full[0] == other.nnz_full[0]
        assert b.vmax == pytest.approx(other.vmax, rel=1e-12)


def test_filters_are_prefixes(e4_file):
    full = two_body_stats(e4_file, 4, 8).blocks("v")
    cut = two_body_stats(e4_file, 4, 8, emax_use=2).blocks("v")
    assert sum(b.nnz_full[0] for b in cut.values()) < sum(b.nnz_full[0] for b in full.values())
    cut2 = two_body_stats(e4_file, 4, 8, e2max_use=4).blocks("v")
    assert sum(b.nnz_full[0] for b in cut2.values()) < sum(b.nnz_full[0] for b in full.values())


def test_manifest_totals_and_provenance(e4_file):
    tint = TInt(16.0, 16)
    stats = two_body_stats(e4_file, 4, 8, tint=tint)
    m = build_manifest(e4_file, 4, 8, None, None, tint, stats, allow_dirty=True)
    json.dumps(m)
    assert len(m["input"]["sha256"]) == 64 and m["git_commit"]
    v = m["parts"]["v"]
    lam = sum(b["lambda_full"] for b in v["blocks"].values())
    assert v["totals"]["all"]["lambda_full"] == pytest.approx(lam, rel=1e-13)
    sec = sum(v["totals"][s]["lambda_full"] for s in ("pp", "pn", "nn"))
    assert sec == pytest.approx(lam, rel=1e-13)
    # structural slots bound the nonzeros, per block and per tau
    for blk in v["blocks"].values():
        assert blk["nnz_full"]["0.0"] <= blk["structural_full"]
        assert blk["nnz_full"]["1e-06"] <= blk["nnz_full"]["5e-09"] <= blk["nnz_full"]["0.0"]
    # T_int adds nonzeros but few new ones; the one-body block exists
    assert m["parts"]["v+tint"]["totals"]["all"]["lambda_full"] > v["totals"]["all"]["lambda_full"]
    assert m["one_body_tint"]["totals"]["lambda_full"] > 0
    assert np.isfinite(m["parts"]["tint"]["totals"]["all"]["max_abs"])


def test_matched_cutoffs():
    from nuham_mscheme.filters import matched_cutoffs

    assert matched_cutoffs(0, 6) == (6, 6, 6)
    assert matched_cutoffs(1, 6) == (7, 8, 9)
    assert matched_cutoffs(1, 8) == (9, 10, 11)


def test_filter_beyond_file_limits_is_an_error(e4_file):
    with pytest.raises(ValueError, match="exceed"):
        two_body_stats(e4_file, 4, 8, emax_use=5)
