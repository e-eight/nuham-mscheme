"""Verify aggregation against dense matrices and HO shell sums; reject mutations."""

import copy
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "tools/comparison/export_snapshot.py"
_spec = importlib.util.spec_from_file_location("export_snapshot", _path)
assert _spec is not None and _spec.loader is not None
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def dense_block(matrix, taus):
    """Independent direct enumeration of full, upper and diagonal positions."""
    n = len(matrix)
    indices = {
        "full": [(i, j) for i in range(n) for j in range(n)],
        "ut": [(i, j) for i in range(n) for j in range(i, n)],
        "diag": [(i, i) for i in range(n)],
    }
    b = {"structural_full": n * n, "structural_ut": n * (n + 1) // 2}
    for name, pairs in indices.items():
        vals = [abs(matrix[i][j]) for i, j in pairs]
        b["lambda_" + name] = math.fsum(vals)
        b["nnz_" + name] = {str(t): sum(v > t for v in vals) for t in taus}
    upper = [abs(matrix[i][j]) for i, j in indices["ut"]]
    b["lambda_ut_above"] = {str(t): math.fsum(v for v in upper if v > t) for t in taus}
    b["max_abs"] = max(abs(v) for row in matrix for v in row)
    return b


def toy():
    """Include exact zeros, signs, a tiny nonzero and a threshold boundary."""
    taus = [0.0, 5e-9, 1.0]
    matrix = [[2.0, -1.0, 0.0], [-1.0, 0.0, 1e-10], [0.0, 1e-10, -3.0]]
    b = dense_block(matrix, taus)
    totals = {
        "lambda_full": 2 * sum(abs(v) for row in matrix for v in row),
        "lambda_ut": 2 * sum(abs(matrix[i][j]) for i in range(3) for j in range(i, 3)),
        "nnz_full": {str(t): 2 * sum(abs(v) > t for row in matrix for v in row) for t in taus},
        "nnz_ut": {
            str(t): 2 * sum(abs(matrix[i][j]) > t for i in range(3) for j in range(i, 3))
            for t in taus
        },
    }
    return {"blocks": {"p:1": b, "n:1": copy.deepcopy(b)}, "totals": totals}, taus, matrix


def test_dense_oracle_and_scaling():
    """Directly scaled matrices independently check projection and sums."""
    part, taus, matrix = toy()
    ex.validate_part(part, taus, 1)
    result = ex.project(part, 1, 16 / 15)["totals"]["all"]
    scaled = [[v * 16 / 15 for v in row] for row in matrix]
    assert result["lambda_full"] == pytest.approx(
        2 * math.fsum(abs(v) for row in scaled for v in row), rel=1e-14
    )
    assert result["nnz_full"] == 2 * sum(v != 0 for row in scaled for v in row)
    assert result["nnz_ut"] == 2 * sum(scaled[i][j] != 0 for i in range(3) for j in range(i, 3))
    assert result["structural_diag"] == 6


@pytest.mark.parametrize("mutation", ["count", "norm", "nan", "sector", "threshold", "total"])
def test_invariants_detect_corruption(mutation):
    """Sabotage checks distinguish valid aggregation from plausible wrong totals."""
    part, taus, _ = toy()
    b = part["blocks"]["p:1"]
    if mutation == "count":
        b["nnz_ut"]["0.0"] += 1
    elif mutation == "norm":
        b["lambda_diag"] *= 1.01
    elif mutation == "nan":
        b["lambda_full"] = float("nan")
    elif mutation == "sector":
        part["blocks"]["pp:1"] = part["blocks"].pop("p:1")
    elif mutation == "threshold":
        b["nnz_ut"]["0.0"] = b["nnz_ut"]["1.0"] - 1
    else:
        part["totals"]["lambda_full"] *= 1.01
    with pytest.raises(ValueError):
        ex.validate_part(part, taus, 1)


def test_combining_coefficients_precedes_absolute_value():
    """An opposite-sign toy term makes adding separate norms observably wrong."""
    v = [[1.0, -2.0], [-2.0, 3.0]]
    t = [[-1.0, 1.0], [1.0, -3.0]]
    combined = [[v[i][j] + t[i][j] for j in range(2)] for i in range(2)]
    joined = dense_block(combined, [0.0])
    assert joined["lambda_full"] == 2.0
    assert joined["nnz_ut"]["0.0"] == 1
    assert joined["lambda_full"] != (
        dense_block(v, [0.0])["lambda_full"] + dense_block(t, [0.0])["lambda_full"]
    )


def test_missing_raw_is_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.load_raw(tmp_path, ex.read(ex.SPEC))


@pytest.fixture(scope="module")
def bundle():
    """Real-input integration requires an explicitly located authenticated bundle."""
    root = os.environ.get("QCHEM_SNAPSHOT_ROOT")
    if root is None:
        pytest.skip("set QCHEM_SNAPSHOT_ROOT for required six-case integration")
    return Path(root)


def shell_sum(emax):
    """Independent HO shell degeneracy sum, without producer orbits/kinetic code."""
    full, upper, diag, count, count_ut = [], [], [], 0, 0
    for l in range(emax + 1):
        degeneracy = 4 * (2 * l + 1)  # both j branches, m, and p/n
        for n in range((emax - l) // 2 + 1):
            for m in range((emax - l) // 2 + 1):
                value = 0.0
                if n == m:
                    value = 8.0 * (2 * n + l + 1.5)
                    diag.append(degeneracy * value)
                elif abs(n - m) == 1:
                    k = max(n, m)
                    value = 8.0 * math.sqrt(k * (k + l + 0.5))
                full.append(degeneracy * value)
                count += degeneracy * (value != 0)
                if n <= m:
                    upper.append(degeneracy * value)
                    count_ut += degeneracy * (value != 0)
    return math.fsum(full), math.fsum(upper), math.fsum(diag), count, count_ut


def test_real_bundle_and_closed_form_kinetic(bundle):
    """HO closed form verifies the kinetic factor independently at all three spaces."""
    data = ex.validate_bundle(bundle)
    for space, emax in (("F8", 8), ("P6", 7), ("P8", 9)):
        b = data["components"][f"NN_{space}/one_body_full"]["totals"]["all"]
        full, upper, diag, count, count_ut = shell_sum(emax)
        for name, value in (("lambda_full", full), ("lambda_ut", upper), ("lambda_diag", diag)):
            assert ex.close(b[name], value)
        assert (b["nnz_full"], b["nnz_ut"]) == (count, count_ut)
    p8 = data["cases"]["P8_E8"]
    assert p8["w_alias_of"] == "F8_E8"
    assert p8["variants"]["full_T"]["components_by_rank"] == [
        "NN_P8/one_body_full",
        "NN_P8/v",
        "W_F8_E8/w",
    ]
    for case in data["cases"].values():
        refs = case["variants"]["intrinsic_T"]["components_by_rank"]
        assert refs[1].endswith("/v+tint")
        assert not any(ref.endswith("/tint") for ref in refs)


@pytest.mark.parametrize(
    "mutation",
    [
        "threshold",
        "alias",
        "kinetic",
        "source_limits",
        "use_limits",
        "norm",
        "raw_hash",
        "missing_field",
        "missing_file",
        "evidence",
        "spec",
    ],
)
def test_bundle_mutations_rejected(bundle, tmp_path, mutation):
    """Even numerically consistent reinterpretations must fail pinned validation."""
    target = tmp_path / "bundle"
    shutil.copytree(bundle, target)
    path = target / "inputs.json"
    data = ex.read(path)
    if mutation == "threshold":
        data["threshold"] = 5e-9
    elif mutation == "alias":
        data["cases"]["P8_E8"]["w_alias_of"] = None
    elif mutation == "kinetic":
        data["cases"]["P6_E8"]["variants"]["intrinsic_T"]["components_by_rank"][1] = "NN_P6/v"
    elif mutation in ("source_limits", "use_limits"):
        field = "file_limits" if mutation == "source_limits" else "filters"
        data["raw_manifests"]["W_P6_E8"]["metadata"][field]["emax"] += 1
    elif mutation == "norm":
        data["components"]["NN_F8/one_body_full"]["totals"]["all"]["lambda_full"] *= 15 / 16
    elif mutation == "raw_hash":
        raw = target / "sources/NN_F8.json"
        d = ex.read(raw)
        d["parts"]["v"]["totals"]["all"]["nnz_ut"]["0.0"] += 1
        raw.write_text(json.dumps(d))
    elif mutation == "missing_field":
        del data["cases"]["F8_E12"]
    elif mutation == "missing_file":
        (target / "sources/W_P6_E8.json").unlink()
    elif mutation == "evidence":
        next((target / "evidence").iterdir()).write_text("changed")
    else:
        (target / "spec.json").write_text("{}")
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        ex.validate_bundle(target)
