"""Export authenticated zero-threshold inputs; never expand matrix elements.

Specification: qchem-to-nuclear ADRs 0007/0008, derivation 10 and the
2026-10-19 input plan. Raw manifests retain diagnostic threshold ladders.
Only zero-threshold counts may be copied when scaling intrinsic one-body T.
"""

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SPEC = Path(__file__).with_name("snapshot-spec-v1.json")
NORMS = ("lambda_full", "lambda_ut", "lambda_diag", "max_abs")
COUNTS = ("nnz_full", "nnz_ut", "nnz_diag")
STRUCT = ("structural_full", "structural_ut", "structural_diag")
SECTORS = {1: {"p", "n"}, 2: {"pp", "pn", "nn"}, 3: {"ppp", "ppn", "pnn", "nnn"}}


def require(ok: bool, message: str) -> None:
    """Reject incomplete or inconsistent inputs."""
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Stream a content hash without loading coefficient files into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read required JSON; missing files are errors."""
    return json.loads(path.read_text())


def write(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace one derived JSON artifact."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def close(a: float, b: float) -> bool:
    """Apply the predeclared summation-grouping tolerance, in MeV."""
    return (
        math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    )


def zero(block: dict[str, Any], scale: float = 1.0) -> dict[str, Any]:
    """Select tau=0 fields; apply positive scaling only to norms and maxima."""
    require(math.isfinite(scale) and scale > 0, "positive finite scale required")
    return {
        **{k: block[k] * scale for k in NORMS},
        **{k: block[k]["0.0"] for k in COUNTS},
        "structural_full": block["structural_full"],
        "structural_ut": block["structural_ut"],
        "structural_diag": 2 * block["structural_ut"] - block["structural_full"],
    }


def aggregate(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce disjoint blocks with exact integers and compensated float sums."""
    require(bool(blocks), "empty component/sector")
    return {
        **{k: sum(b[k] for b in blocks) for k in (*COUNTS, *STRUCT)},
        **{k: math.fsum(b[k] for b in blocks) for k in NORMS if k != "max_abs"},
        "max_abs": max(b["max_abs"] for b in blocks),
    }


def validate_part(part: dict[str, Any], taus: list[float], rank: int) -> dict[str, float]:
    """Check Hermitian identities, capacities and every diagnostic threshold."""
    keys = [str(t) for t in taus]
    require(
        bool(keys)
        and keys[0] == "0.0"
        and all(math.isfinite(t) and t >= 0 for t in taus)
        and taus == sorted(set(taus)),
        "threshold order",
    )
    blocks = part["blocks"]
    require({k.split(":")[0] for k in blocks} == SECTORS[rank], "block sector coverage")
    residuals = {"max_norm_identity_abs_MeV": 0.0, "max_total_relative_residual": 0.0}
    for name, b in blocks.items():
        require(all(type(b[k]) is int for k in STRUCT[:2]), "structural integer")
        diag = 2 * b["structural_ut"] - b["structural_full"]
        require(0 <= diag <= b["structural_ut"] <= b["structural_full"], "structural bounds")
        require(all(math.isfinite(b[k]) and b[k] >= 0 for k in NORMS), "invalid norm")
        target = 2 * b["lambda_ut"] - b["lambda_diag"]
        require(close(b["lambda_full"], target), "norm identity " + name)
        residuals["max_norm_identity_abs_MeV"] = max(
            residuals["max_norm_identity_abs_MeV"], abs(b["lambda_full"] - target)
        )
        require(
            b["lambda_diag"] <= b["lambda_ut"] <= b["lambda_full"]
            and b["max_abs"] <= b["lambda_full"],
            "norm bounds",
        )
        for field, capacity in zip(
            COUNTS, (b["structural_full"], b["structural_ut"], diag), strict=True
        ):
            require(set(b[field]) == set(keys), "threshold keys")
            vals = [b[field][t] for t in keys]
            require(all(type(v) is int and 0 <= v <= capacity for v in vals), "count bounds")
            require(all(a >= c for a, c in zip(vals, vals[1:], strict=False)), "count order")
        require(set(b["lambda_ut_above"]) == set(keys), "norm threshold keys")
        above = [b["lambda_ut_above"][t] for t in keys]
        require(all(math.isfinite(v) and v >= 0 for v in above), "threshold norm")
        require(close(above[0], b["lambda_ut"]), "zero-threshold norm")
        require(
            all(a >= c or close(a, c) for a, c in zip(above, above[1:], strict=False)),
            "norm threshold order",
        )
        for t in keys:
            require(b["nnz_full"][t] == 2 * b["nnz_ut"][t] - b["nnz_diag"][t], "count identity")
            require(
                b["nnz_ut"][t] > 0 or b["lambda_ut_above"][t] == 0, "empty support has nonzero norm"
            )
    totals = {"all": part["totals"]} if rank == 1 else part["totals"]
    require(
        set(totals) == ({"all"} if rank == 1 else SECTORS[rank] | {"all"}), "total sector coverage"
    )
    fields = ("lambda_full", "lambda_ut", "nnz_full", "nnz_ut")
    if rank != 1:
        fields += ("structural_ut", "max_abs", "lambda_ut_above")
    for sector, total in totals.items():
        selected = [b for k, b in blocks.items() if sector == "all" or k.split(":")[0] == sector]
        require(set(total) == set(fields), "total fields")
        for field in fields:
            for tau in keys if field in (*COUNTS, "lambda_ut_above") else [None]:
                if tau is not None:
                    require(set(total[field]) == set(keys), "total threshold keys")
                actual = total[field][tau] if tau is not None else total[field]
                values = [b[field][tau] if tau is not None else b[field] for b in selected]
                integer = field in (*COUNTS, *STRUCT)
                expected = (
                    max(values)
                    if field == "max_abs"
                    else sum(values)
                    if integer
                    else math.fsum(values)
                )
                require(
                    (type(actual) is int and actual == expected)
                    if integer
                    else close(actual, expected),
                    "total " + sector + "/" + field,
                )
                if not integer:
                    residuals["max_total_relative_residual"] = max(
                        residuals["max_total_relative_residual"],
                        abs(actual - expected) / max(1.0, abs(expected)),
                    )
    return residuals


def project(part: dict[str, Any], rank: int, scale: float = 1.0) -> dict[str, Any]:
    """Export zero fields by block and sector, deriving missing diagonal totals."""
    blocks = {name: zero(b, scale) for name, b in part["blocks"].items()}
    totals = {
        sector: aggregate(
            [b for k, b in blocks.items() if sector == "all" or k.split(":")[0] == sector]
        )
        for sector in sorted(SECTORS[rank] | {"all"})
    }
    return {"rank": rank, "norm_scale": scale, "blocks": blocks, "totals": totals}


def load_raw(directory: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Authenticate bytes and metadata against the separately Git-pinned spec."""
    docs = {}
    for name, item in spec["manifests"].items():
        path = directory / (name + ".json")
        require(sha(path) == item["sha256"], "manifest hash " + name)
        d = read(path)
        for key, expected in item["metadata"].items():
            actual = d["input"]["sha256"] if key == "input_sha256" else d[key]
            require(actual == expected, "metadata " + name + "/" + key)
        require(
            set(d["parts"]) == ({"v", "tint", "v+tint"} if name.startswith("NN") else {"w"}),
            "component coverage",
        )
        docs[name] = d
    return docs


def build(docs: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Build the six cases, keeping rank-two coefficient sums intact."""
    components, checks = {}, {}
    require(set(docs) == set(spec["manifests"]), "manifest coverage")
    for name, d in docs.items():
        parts = dict(d["parts"])
        if name.startswith("NN"):
            parts["one_body_tint"] = d["one_body_tint"]
        for part_name, part in parts.items():
            rank = 1 if part_name == "one_body_tint" else 3 if part_name == "w" else 2
            key = name + "/" + part_name
            checks[key] = validate_part(part, d["taus"], rank)
            components[key] = dict(project(part, rank), raw_manifest=name, raw_field=part_name)
            if rank == 1:
                components[name + "/one_body_full"] = dict(
                    project(part, 1, 16 / 15),
                    raw_manifest=name,
                    raw_field=part_name,
                    conversion="A/(A-1), A=16; zero support unchanged",
                )
    cases = {}
    for name, definition in spec["cases"].items():
        nn, w = definition["nn_manifest"], definition["w_manifest"] + "/w"
        variants = {}
        for variant, one, two in (
            ("full_T", "one_body_full", "v"),
            ("intrinsic_T", "one_body_tint", "v+tint"),
        ):
            refs = [nn + "/" + one, nn + "/" + two, w]
            totals = [components[k]["totals"]["all"] for k in refs]
            variants[variant] = {
                "components_by_rank": refs,
                "lambda_full": math.fsum(b["lambda_full"] for b in totals),
                "loaded_upper_count": sum(b["nnz_ut"] for b in totals),
            }
        cases[name] = dict(definition, variants=variants)
    return {
        **{
            k: spec[k]
            for k in (
                "schema",
                "units",
                "threshold",
                "loading_policy",
                "common",
                "limitations",
                "aggregation",
            )
        },
        "spec_sha256": sha(SPEC),
        "sources": spec["sources"],
        "raw_manifests": spec["manifests"],
        "evidence": spec["evidence"],
        "components": components,
        "cases": cases,
        "checks": checks,
    }


def validate_bundle(directory: Path) -> dict[str, Any]:
    """Reconstruct the view from pinned raw bytes; reject altered case semantics."""
    spec = read(SPEC)
    require(sha(directory / "spec.json") == sha(SPEC), "spec hash")
    for name, item in spec["evidence"].items():
        require(sha(directory / "evidence" / name) == item["sha256"], "evidence hash")
    expected = build(load_raw(directory / "sources", spec), spec)
    require(read(directory / "inputs.json") == expected, "snapshot differs from pinned inputs")
    return expected


def git(repo: Path, *args: str) -> str:
    """Run a checked, read-only Git query."""
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def export(inputs: Path, science: Path, out: Path, commit: str) -> None:
    """Write a provenance manifest before validating and exporting local inputs."""
    repo = Path(__file__).resolve().parents[2]
    spec = read(SPEC)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest: dict[str, Any] = dict(
        status="running",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git={
            "commit": git(repo, "rev-parse", "HEAD"),
            "branch": git(repo, "branch", "--show-current"),
            "dirty": bool(git(repo, "status", "--porcelain")),
            "diff": git(repo, "diff", "HEAD"),
        },
        params={
            "spec_sha256": sha(SPEC),
            "norm_rtol": 1e-12,
            "norm_atol_MeV": 1e-12,
            "inputs_root": str(inputs),
            "science_root": str(science),
            "threshold": 0.0,
            "restart": "new output directory; no checkpoint",
        },
        seeds={},
        env={
            "python": sys.version,
            "platform": platform.platform(),
            "host": platform.node(),
            "packages": "stdlib only",
            "compiler": platform.python_compiler(),
            "blas": "not used",
            "threads": 1,
            "gpu": False,
        },
        job_id=None,
        inputs=[],
        outputs=[],
    )
    write(out / "manifest.json", manifest)
    try:
        require(
            len(commit) == 40
            and manifest["git"]["commit"] == commit
            and not manifest["git"]["dirty"],
            "clean pinned exporter required",
        )
        legacy = science / "checks/nuham_inputs.py"
        require(sha(legacy) == spec["historical_snapshot_sha256"], "historical snapshot changed")
        manifest["historical_snapshot_sha256"] = sha(legacy)
        for digest, source in spec["sources"].items():
            path = inputs / source["path"]
            require(path.stat().st_size == source["bytes"] and sha(path) == digest, "source hash")
            manifest["inputs"].append(
                {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
            )
        for group, target in (("manifests", "sources"), ("evidence", "evidence")):
            (out / target).mkdir()
            for name, item in spec[group].items():
                root = inputs if item.get("root") == "inputs" else science
                path = root / item["path"]
                require(sha(path) == item["sha256"], "input hash " + name)
                dest = out / target / (name + ".json" if group == "manifests" else name)
                shutil.copyfile(path, dest)
                require(sha(dest) == item["sha256"], "copy hash " + name)
                manifest["inputs"].append({"path": str(path), "sha256": item["sha256"]})
        shutil.copyfile(SPEC, out / "spec.json")
        result = build(load_raw(out / "sources", spec), spec)
        write(out / "inputs.json", result)
        validate_bundle(out)
        require(sha(legacy) == spec["historical_snapshot_sha256"], "historical snapshot changed")
        require(
            git(repo, "rev-parse", "HEAD") == commit and not git(repo, "status", "--porcelain"),
            "exporter changed during export",
        )
        manifest.update(status="passed", exit_code=0, validation="pinned raw reconstruction passed")
    except BaseException as exc:
        manifest.update(status="failed", exit_code=1, error=repr(exc))
        raise
    finally:
        manifest["wall_time_s"] = time.monotonic() - start
        manifest["outputs"] = [
            {"path": str(p.relative_to(out)), "sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*"))
            if p.is_file() and p != out / "manifest.json"
        ]
        write(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "wall_time_s": manifest["wall_time_s"]}))


def main() -> None:
    """Export from clean pinned code, or validate an already exported bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    for name in ("inputs", "science", "out"):
        ex.add_argument("--" + name, type=Path, required=True)
    ex.add_argument("--commit", required=True)
    va = sub.add_parser("validate")
    va.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        validate_bundle(args.directory)
        print("passed: six cases reconstructed from authenticated raw manifests")
    else:
        export(args.inputs.resolve(), args.science.resolve(), args.out.resolve(), args.commit)


if __name__ == "__main__":
    main()
