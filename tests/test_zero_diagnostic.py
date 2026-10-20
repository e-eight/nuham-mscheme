"""Check the diagnostic by invariants, round trips and deliberate corruption."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "diagnose_zeros3", Path(__file__).resolve().parents[1] / "tools/diagnose_zeros3.py"
)
assert SPEC is not None and SPEC.loader is not None
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


@pytest.fixture(scope="module")
def fixture():
    return d.native_fixture()


def test_fixture_round_trip_and_rejections(fixture, tmp_path):
    path = tmp_path / "fixture.npz"
    np.savez_compressed(path, **fixture)
    assert d.fingerprints(d.load(path)) == d.fingerprints(fixture)
    for field, update in (
        ("offsets", lambda x: x.__setitem__(1, 0)),
        ("keys", lambda x: x.__setitem__((0, 0), 123)),
        ("values", lambda x: x.__setitem__(0, np.nan)),
    ):
        bad = copy.deepcopy(fixture)
        update(bad[field])
        with pytest.raises(ValueError):
            d.unpack(bad)


def test_batched_assembly_and_trace(fixture, tmp_path):
    result = d.evaluate(fixture, tmp_path / "evaluation")
    assert result["direct_count_checks"].startswith("all blocks")
    tuples = d.unpack(fixture)
    basis = d.kernel.TripleBasis(d.Layout(*d.LIMITS).orbits)
    # Trace a nontrivial off-diagonal tuple even when no zeros differ locally.
    tid = next(
        i for i, t in enumerate(tuples) if t.bra != t.ket and np.count_nonzero(t.values) > 10
    )
    d.trace_tuple(basis, tuples[tid], tmp_path / "trace.npz")
    with np.load(tmp_path / "trace.npz") as z:
        assert z["batch_valid"].any()
        assert any(k.startswith("s_") for k in z.files)
    meta = json.loads((tmp_path / "trace.json").read_text())
    assert all(v for k, v in meta.items() if k.endswith("replay_equal"))


def test_accumulator_corruption_is_detected(fixture, tmp_path, monkeypatch):
    original = d.ThreeBodyStats.blocks

    def corrupted(self):
        blocks = original(self)
        next(iter(blocks.values())).nnz_full[0] += 1
        return blocks

    monkeypatch.setattr(d.ThreeBodyStats, "blocks", corrupted)
    with pytest.raises(ValueError, match="accumulator/direct"):
        d.evaluate(fixture, tmp_path / "bad")


def test_zero_mismatch_capture(fixture, tmp_path, monkeypatch):
    """An injected symmetric zero must produce two indexed entries and a trace."""
    original = d.dense_blocks

    def corrupted(*args):
        blocks = original(*args)
        for matrix, _ in blocks.values():
            candidates = np.argwhere(np.triu(np.abs(matrix), 1) > 0.1)
            if len(candidates):
                r, c = candidates[0]
                matrix[r, c] = matrix[c, r] = 0.0
                break
        return blocks

    monkeypatch.setattr(d, "dense_blocks", corrupted)
    out = tmp_path / "zero-mutation"
    result = d.evaluate(fixture, out)
    assert result["zero_disagreements"] == 2
    entries = json.loads((out / "mismatches.json").read_text())
    assert all(e["dense"] == 0 and abs(e["batched"]) > 0.1 for e in entries)
    assert all(e["mirror_disagrees"] for e in entries)
    assert list(out.glob("trace-*.npz"))
