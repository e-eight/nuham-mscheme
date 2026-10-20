"""One-off orchestration/verification of existing run3; no new expansion algorithm.

Independent small oracles live in the pinned producer tests. Full W8 comparison
is a cross-platform regression, not an independent physical-accuracy oracle.
Bookkeeping follows derivation 10 and the accepted step-3 input plan.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

BUNDLE = Path(__file__).resolve().parent
SOURCE_SHA = "c347cbcfe5c2095b99bed6659243a112fbbbaf8af4a7c645b6888e3a3e89e552"
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def require(ok: bool, message: str) -> None:
    """Fail closed on missing provenance or a numerical mismatch."""
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture(cmd: list[str]) -> dict[str, Any]:
    """Capture a small metadata command."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"command": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def close(a: float, b: float) -> bool:
    """Use the accepted cross-environment regression tolerance in MeV."""
    return (
        math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    )


def differences(a: object, b: object, path: str = "parts") -> list[dict[str, Any]]:
    """All integer counts exact; all norm/max fields use the declared tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [{"path": path, "error": "keys differ"}]
        return [v for k in a for v in differences(a[k], b[k], path + "/" + k)]
    same = (
        (type(b) is int and a == b)
        if type(a) is int
        else (isinstance(a, (int, float)) and isinstance(b, (int, float)) and close(a, b))
    )
    return [] if same else [{"path": path, "reference": a, "actual": b}]


def invariants(doc: dict[str, Any]) -> None:
    """Verify derivation 10 count/norm identities and threshold bounds."""
    taus = [str(t) for t in doc["taus"]]
    require(
        taus[0] == "0.0" and list(map(float, taus)) == sorted(map(float, taus)), "threshold order"
    )
    part = doc["parts"]["w"]
    blocks = part["blocks"]
    require(bool(blocks), "empty blocks")
    require(set(part["totals"]) == {"ppp", "ppn", "pnn", "nnn", "all"}, "sector coverage")
    for name, b in blocks.items():
        require(name.split(":")[0] in {"ppp", "ppn", "pnn", "nnn"}, "unknown sector")
        diag = 2 * b["structural_ut"] - b["structural_full"]
        require(
            0 <= diag <= b["structural_ut"] <= b["structural_full"], "structural identity " + name
        )
        for f in ("lambda_full", "lambda_ut", "lambda_diag", "max_abs"):
            require(math.isfinite(b[f]) and b[f] >= 0, "invalid norm " + name)
        require(
            close(b["lambda_full"], 2 * b["lambda_ut"] - b["lambda_diag"]), "norm identity " + name
        )
        require(
            b["lambda_diag"] <= b["lambda_ut"] <= b["lambda_full"]
            and b["max_abs"] <= b["lambda_full"],
            "norm bounds " + name,
        )
        for f, capacity in (
            ("nnz_full", b["structural_full"]),
            ("nnz_ut", b["structural_ut"]),
            ("nnz_diag", diag),
        ):
            require(set(b[f]) == set(taus), "threshold keys " + name)
            ladder = [b[f][t] for t in taus]
            require(
                all(type(x) is int and 0 <= x <= capacity for x in ladder), "count bounds " + name
            )
            require(
                all(x >= y for x, y in zip(ladder, ladder[1:], strict=False)),
                "count monotonicity " + name,
            )
        require(set(b["lambda_ut_above"]) == set(taus), "threshold norm keys")
        above = [b["lambda_ut_above"][t] for t in taus]
        require(all(math.isfinite(x) and x >= 0 for x in above), "threshold norm finite")
        require(
            all(x >= y or close(x, y) for x, y in zip(above, above[1:], strict=False)),
            "threshold norm order",
        )
        require(close(above[0], b["lambda_ut"]), "zero-threshold norm")
        for t in taus:
            require(
                b["nnz_full"][t] == 2 * b["nnz_ut"][t] - b["nnz_diag"][t], "count identity " + name
            )
    for sector, total in part["totals"].items():
        selected = [
            b for key, b in blocks.items() if sector == "all" or key.split(":")[0] == sector
        ]
        require(bool(selected), "missing sector " + sector)
        require(
            total["structural_ut"] == sum(b["structural_ut"] for b in selected), "structural sum"
        )
        for f in ("lambda_full", "lambda_ut"):
            require(close(total[f], math.fsum(b[f] for b in selected)), "norm sum " + sector)
        require(close(total["max_abs"], max(b["max_abs"] for b in selected)), "maximum " + sector)
        for f in ("nnz_full", "nnz_ut", "lambda_ut_above"):
            require(set(total[f]) == set(taus), "total threshold keys")
            for t in taus:
                vals = [b[f][t] for b in selected]
                require(
                    close(total[f][t], math.fsum(vals))
                    if f == "lambda_ut_above"
                    else total[f][t] == sum(vals),
                    "total " + sector + "/" + f + "/" + t,
                )


def main() -> None:
    """Run the fail-stop compatibility ladder from a clean pinned checkout."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pin = a.commit
    require(
        len(pin) == 40 and all(c in "0123456789abcdef" for c in pin), "full Git commit required"
    )
    repo, source, out = a.repo.resolve(), a.source.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    m = dict(
        status="running",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=platform.node(),
        architecture=platform.machine(),
        python=sys.version,
        job={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        seeds={"synthetic_operator": 20261010, "explicit_sympy_queries": 7, "reader_ranges": 5},
        params=dict(
            file_limits=[8, 16, 8],
            smoke_use=[2, 4, 4],
            baseline_use=[8, 16, 8],
            smoke_workers=[1, 3],
            baseline_workers=3,
            n_chunks=256,
            verify_count=True,
            rtol=1e-12,
            atol_MeV=1e-12,
            restart="from start in new directory; no checkpoint",
            P6=False,
        ),
        inputs=[],
        stages=[],
        outputs=[],
    )

    def save() -> None:
        tmp = out / "manifest.json.tmp"
        tmp.write_text(json.dumps(m, indent=2) + "\n")
        tmp.replace(out / "manifest.json")

    save()

    def interrupted(signum: int, frame: object) -> None:
        raise RuntimeError("interrupted by signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            p = BUNDLE / name
            require(sha(p) == expected, "bundle hash " + name)
            m["inputs"].append(dict(path=str(p), sha256=expected))
        for line in (repo / "tools/fixtures/SHA256SUMS").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            path = repo / name
            require(sha(path) == expected, "fixture hash " + name)
            m["inputs"].append(dict(path=str(path), sha256=expected))
        env = os.environ.copy()
        env.update({k: "1" for k in THREADS})
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONHASHSEED="0",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            QCHEM_W8_SOURCE=str(source),
            PYTHONPATH=str(repo / "src") + os.pathsep + str(BUNDLE),
        )
        env.pop("PYTHONOPTIMIZE", None)
        m["threads"] = {k: env[k] for k in THREADS}

        def clean() -> None:
            commit = capture(["git", "-C", str(repo), "rev-parse", "HEAD"])
            status = capture(["git", "-C", str(repo), "status", "--porcelain"])
            require(commit["returncode"] == status["returncode"] == 0, "git metadata unavailable")
            require(
                commit["stdout"].strip() == pin and not status["stdout"].strip(),
                "producer must be clean " + pin,
            )

        m["git"] = {
            "commit": capture(["git", "-C", str(repo), "rev-parse", "HEAD"]),
            "status": capture(["git", "-C", str(repo), "status", "--porcelain"]),
            "diff": capture(["git", "-C", str(repo), "diff", "HEAD"]),
        }
        clean()
        kernel_diff = capture(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--exit-code",
                "c69f815c15411b4a1ac16802aff6e2d99967baa5",
                pin,
                "--",
                "src",
            ]
        )
        require(kernel_diff["returncode"] == 0, "numerical producer changed from diagnosed base")
        m["kernel_base"] = "c69f815c15411b4a1ac16802aff6e2d99967baa5"
        require(sha(source) == SOURCE_SHA, "W8 source hash differs")
        m["inputs"].append(dict(path=str(source), sha256=SOURCE_SHA, bytes=source.stat().st_size))
        m["producer_files"] = {
            str(p.relative_to(repo)): sha(p)
            for root in (repo / "src", repo / "tests")
            for p in sorted(root.rglob("*.py"))
        }
        m["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        m["hardware"] = capture(["lscpu"])
        m["compiler"] = capture(["cc", "--version"])
        os.environ.update({k: "1" for k in THREADS})
        sys.path.insert(0, str(repo / "src"))
        import numpy as np

        from nuham_mscheme import run3

        require(
            Path(run3.__file__).resolve() == repo / "src/nuham_mscheme/run3.py",
            "wrong imported producer",
        )
        require(run3.N_CHUNKS == 256, "chunk partition changed")
        m["params"]["taus"] = list(run3.TAUS)
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        m["blas"] = config.getvalue()
        save()

        def run(name: str, command: list[str]) -> None:
            clean()
            stage = dict(name=name, command=command, status="running")
            m["stages"].append(stage)
            save()
            tick = time.monotonic()
            with (out / (name + ".log")).open("w") as log:
                p = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            stage.update(
                exit_code=p.returncode,
                wall_time_s=time.monotonic() - tick,
                max_child_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                memory_scope="cumulative largest child, not aggregate simultaneous RSS",
                status="passed" if p.returncode == 0 else "failed",
            )
            save()
            require(p.returncode == 0, name + " failed; see log")

        run(
            "small-tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "input_paths",
                "tests/test_me3j.py",
                "tests/test_mscheme3.py",
                "tests/test_run3.py",
                "tests/test_zero_diagnostic.py",
                "tests/test_zero_oracle.py",
                "--basetemp=" + str(out / "pytest-tmp"),
                "--junitxml=" + str(out / "tests.xml"),
            ],
        )
        cases = ET.parse(out / "tests.xml").findall(".//testcase")
        require(
            len(cases) == 50 and all(len(c) == 0 for c in cases),
            "expected 50 passing tests and zero skips",
        )

        def sweep(name: str, use: tuple[int, int, int], workers: int) -> dict[str, Any]:
            target = out / (name + ".json")
            cmd = [
                sys.executable,
                "-m",
                "nuham_mscheme.run3",
                str(source),
                "--emax",
                "8",
                "--e2max",
                "16",
                "--e3max",
                "8",
                "--emax-use",
                str(use[0]),
                "--e2max-use",
                str(use[1]),
                "--e3max-use",
                str(use[2]),
                "--workers",
                str(workers),
                "--verify-count",
                "--out",
                str(target),
            ]
            run(name, cmd)
            d = json.loads(target.read_text())
            require(d["git_commit"] == pin and d["git_dirty"] is False, "output producer")
            require(
                d["input"]["sha256"] == SOURCE_SHA and Path(d["input"]["path"]) == source,
                "output source",
            )
            require(d["file_limits"] == dict(emax=8, e2max=16, e3max=8), "output layout")
            require(
                d["filters"] == dict(zip(("emax", "e2max", "e3max"), use, strict=True)),
                "output filters",
            )
            require(
                d["n_chunks"] == 256 and d["taus"] == m["params"]["taus"],
                "output thresholds/chunks",
            )
            invariants(d)
            return d

        one = sweep("smoke-w1", (2, 4, 4), 1)
        three = sweep("smoke-w3", (2, 4, 4), 3)
        require(one["parts"] == three["parts"], "smoke worker dependence")
        smoke_ref = json.loads((BUNDLE / "reference-smoke.json").read_text())
        smoke_diff = differences(smoke_ref["parts"], one["parts"])
        (out / "smoke-differences.json").write_text(json.dumps(smoke_diff, indent=2) + "\n")
        require(not smoke_diff, "smoke differs from local reference")
        baseline = sweep("baseline", (8, 16, 8), 3)
        ref = json.loads((BUNDLE / "reference-baseline.json").read_text())
        delta = differences(ref["parts"], baseline["parts"])
        (out / "baseline-differences.json").write_text(json.dumps(delta, indent=2) + "\n")
        require(not delta, "W8 differs from local reference; do not proceed to P6")
        require(
            baseline["parts"]["w"]["totals"]["all"]["nnz_ut"]["0.0"] == 526367562,
            "W8 zero-count gate",
        )
        clean()
        require(sha(source) == SOURCE_SHA, "source changed during job")
        require(
            m["producer_files"]
            == {
                str(p.relative_to(repo)): sha(p)
                for root in (repo / "src", repo / "tests")
                for p in sorted(root.rglob("*.py"))
            },
            "producer changed during job",
        )
        m.update(
            status="passed",
            exit_code=0,
            test_count=len(cases),
            worker_results_exact=True,
            cross_platform_count_differences=0,
            comparison="all blocks/totals/thresholds, norm tolerance 1e-12",
            nnz_ut_zero=526367562,
        )
    except BaseException as exc:
        m.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        m["wall_time_s"] = time.monotonic() - start
        m["outputs"] = [
            dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "manifest.json"
        ]
        save()
        print(json.dumps({k: m[k] for k in ("status", "wall_time_s", "exit_code")}))


if __name__ == "__main__":
    main()
