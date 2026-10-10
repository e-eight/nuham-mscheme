"""Tests of the three-body sweep driver."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_me3j import _text

from nuham_mscheme.me3j import filter_me3j
from nuham_mscheme.run3 import build_manifest, chunk_rows, main, three_body_stats

LIMITS = (3, 6, 6)


@pytest.fixture(scope="module")
def path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("run3") / "t.me3j"
    p.write_bytes(_text(*LIMITS))
    return p


def _same(a, b, exact: bool) -> None:
    ba, bb = a.blocks(), b.blocks()
    assert ba.keys() == bb.keys()
    for k in ba:
        if exact:
            assert ba[k] == bb[k]  # bitwise, including the 1-norm sums
        else:
            assert (
                ba[k].nnz_full == bb[k].nnz_full and ba[k].structural_full == bb[k].structural_full
            )
            assert ba[k].lambda_full == pytest.approx(bb[k].lambda_full, rel=1e-12)
            assert ba[k].vmax == bb[k].vmax


def test_chunk_rows_partition():
    b = chunk_rows(LIMITS, LIMITS, 16)
    assert b[0] == 0 and b == sorted(b) and len(b) == 17
    assert chunk_rows(LIMITS, LIMITS, 16) == b
    assert chunk_rows(LIMITS, (2, 4, 4), 16)[-1] == b[-1]  # the whole file is covered


def test_workers_do_not_change_result(path):
    a = three_body_stats(path, LIMITS, workers=1, n_chunks=16)
    b = three_body_stats(path, LIMITS, workers=3, n_chunks=16)
    _same(a, b, exact=True)


def test_chunk_count_does_not_change_the_counts(path):
    a = three_body_stats(path, LIMITS, n_chunks=4)
    b = three_body_stats(path, LIMITS, n_chunks=32)
    _same(a, b, exact=False)


def test_filter_equals_the_cut_file(path, tmp_path):
    use = (2, 4, 4)
    cut = tmp_path / "cut.me3j"
    filter_me3j(path, cut, (6, 6), use)
    _same(
        three_body_stats(path, LIMITS, use, n_chunks=8),
        three_body_stats(cut, use, n_chunks=8),
        False,
    )
    with pytest.raises(ValueError, match="exceed"):
        three_body_stats(path, LIMITS, (3, 6, 7))


def test_manifest_totals_and_provenance(path, tmp_path):
    stats = three_body_stats(path, LIMITS, n_chunks=8)
    m = build_manifest(path, LIMITS, LIMITS, stats, allow_dirty=True)
    json.dumps(m)
    assert len(m["input"]["sha256"]) == 64 and m["git_commit"]
    w = m["parts"]["w"]
    lam = sum(b["lambda_full"] for b in w["blocks"].values())
    assert w["totals"]["all"]["lambda_full"] == pytest.approx(lam, rel=1e-13)
    sec = sum(w["totals"][s]["lambda_full"] for s in ("ppp", "ppn", "pnn", "nnn"))
    assert sec == pytest.approx(lam, rel=1e-13)
    for blk in w["blocks"].values():
        assert blk["nnz_full"]["0.0"] <= blk["structural_full"]
        assert blk["nnz_full"]["1e-06"] <= blk["nnz_full"]["5e-09"] <= blk["nnz_full"]["0.0"]
    assert np.isfinite(w["totals"]["all"]["max_abs"])
    out = tmp_path / "m.json"
    main(
        [
            str(path),
            "--emax",
            "3",
            "--e2max",
            "6",
            "--e3max",
            "6",
            "--nmax",
            "2",
            "--e-fermi",
            "0",
            "--out",
            str(out),
            "--allow-dirty",
        ]
    )
    assert json.loads(out.read_text())["filters"] == {"emax": 2, "e2max": 2, "e3max": 2}


def test_dirty_tree_fails_at_the_start(monkeypatch, tmp_path):
    """A dirty tree stops the driver before any work, unless --allow-dirty is given."""
    from nuham_mscheme import run2

    monkeypatch.setattr(run2, "_git", lambda *a: " M file.py")
    with pytest.raises(RuntimeError, match="dirty"):
        main(["x.gz", "--emax", "2", "--e2max", "2", "--e3max", "2", "--out", str(tmp_path / "o")])
    run2.require_clean_tree(True)
