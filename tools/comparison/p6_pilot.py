"""Bounded P6 pilot with the verified W8 gate and disjoint-support reconstruction.

Bookkeeping follows qchem-to-nuclear derivation 10, eq:comparison-mask-order
and eq:comparison-consistent-d-lambda. No expansion kernel is changed here.
"""

from __future__ import annotations

# Select the pinned checkout before importing numerical modules.
# ruff: noqa: E402
import argparse
import contextlib
import importlib.metadata
import io
import itertools
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from compatibility import SOURCE_SHA, THREADS, capture, close, differences, invariants, require, sha

if TYPE_CHECKING:
    from nuham_mscheme.stats3 import ThreeBodyStats

GATE_COMMIT = "be6b584c27c875c91dc58128a2255f69ee309b29"
GATE_SHA = "483b8d4c65b474ec77406d599201042bb4394516e4edaa569c8e4ced4508e01c"
LIMITS = (8, 16, 8)
USE = (7, 8, 8)


def retained(energies: tuple[int, ...], use: tuple[int, int, int]) -> bool:
    """Test all one/two/three-slot energy bounds without the producer keep mask."""
    return all(
        sum(subset) <= bound
        for rank, bound in enumerate(use, 1)
        for subset in itertools.combinations(energies, rank)
    )


def discarded_stats(
    path: Path, limits: tuple[int, int, int], use: tuple[int, int, int]
) -> tuple[ThreeBodyStats, dict[str, int]]:
    """Expand only tuples with at least one excluded endpoint, in the full basis."""
    from nuham_mscheme.me3j import iter_tuples3
    from nuham_mscheme.mscheme3 import TripleBasis
    from nuham_mscheme.orbits import orbit_list
    from nuham_mscheme.stats3 import ThreeBodyStats

    basis = TripleBasis(orbit_list(limits[0]))
    stats = ThreeBodyStats(basis)
    seen = dropped = 0
    for tup in iter_tuples3(path, *limits):
        seen += 1
        bra = retained(tuple(basis.orbits[i].e for i in tup.bra), use)
        ket = retained(tuple(basis.orbits[i].e for i in tup.ket), use)
        if not (bra and ket):
            stats.add_tuple(tup)
            dropped += 1
    return stats, dict(source_tuples=seen, excluded_tuples=dropped)


def parts(stats: ThreeBodyStats) -> dict[str, Any]:
    """Use the producer's established serialization for a separately selected support."""
    from nuham_mscheme.run3 import _block_dict, _totals

    blocks = stats.blocks()
    return dict(
        w=dict(
            blocks={f"{s}:{m}": _block_dict(b) for (s, m), b in sorted(blocks.items())},
            totals=_totals(blocks),
        )
    )


def reconstruct(full: dict, kept: dict, removed: dict, path: str = "parts") -> None:
    """Require exact additive counts and tolerant additive norms on disjoint supports.

    Zero-valued absent blocks are allowed; an unexpected block/field is not.
    Absolute-value sums add because no coefficient belongs to both supports.
    This is a bookkeeping oracle, not independent coupling algebra.
    """
    require(set(kept) | set(removed) == set(full), "support keys differ")
    for key, value in full.items():
        a, b = (
            kept.get(key, {} if isinstance(value, dict) else 0),
            removed.get(key, {} if isinstance(value, dict) else 0),
        )
        if isinstance(value, dict):
            reconstruct(value, a, b, path + "/" + key)
        else:
            expected = max(a, b) if key == "max_abs" else a + b
            if type(value) is int:
                require(
                    type(expected) is int and expected == value,
                    "complement count " + path + "/" + key,
                )
            else:
                require(close(value, expected), "complement norm " + path + "/" + key)


def gate(path: Path) -> dict[str, Any]:
    """Authenticate the successful remote prerequisite and every recorded output."""
    require(sha(path) == GATE_SHA, "compatibility manifest hash")
    doc = json.loads(path.read_text())
    require(doc["status"] == "passed" and doc["exit_code"] == 0, "compatibility failed")
    require(doc["test_count"] == 50, "compatibility test coverage")
    require(doc["git"]["commit"]["stdout"].strip() == GATE_COMMIT, "compatibility revision")
    require(
        len(doc["stages"]) == 4 and all(s["status"] == "passed" for s in doc["stages"]),
        "compatibility stages",
    )
    for item in doc["outputs"]:
        artifact = path.parent / item["path"]
        require(
            artifact.stat().st_size == item["bytes"] and sha(artifact) == item["sha256"],
            "compatibility output " + item["path"],
        )
    return doc


def main() -> None:
    """Record provenance, smoke the existing driver, then compute the pilot once."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()
    require(len(a.commit) == 40 and all(c in "0123456789abcdef" for c in a.commit), "full SHA")
    source, gate_path, out = a.source.resolve(), a.gate.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    m: dict[str, Any] = dict(
        status="running",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git=dict(
            requested_commit=a.commit,
            commit=capture(["git", "-C", str(ROOT), "rev-parse", "HEAD"])["stdout"].strip(),
            branch=capture(["git", "-C", str(ROOT), "branch", "--show-current"])["stdout"].strip(),
            dirty=bool(
                capture(["git", "-C", str(ROOT), "status", "--porcelain"])["stdout"].strip()
            ),
            diff=capture(["git", "-C", str(ROOT), "diff", "HEAD"])["stdout"],
        ),
        host=platform.node(),
        architecture=platform.machine(),
        python=sys.version,
        job={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        params=dict(
            file_limits=LIMITS,
            pilot_use=USE,
            smoke_use=(2, 4, 4),
            workers=3,
            smoke_workers=[1, 3],
            n_chunks=256,
            verify_count=True,
            smoke_only=a.smoke_only,
            rtol=1e-12,
            atol_MeV=1e-12,
            restart="none; restart from beginning in new directory",
        ),
        seeds={},
        inputs=[],
        stages=[],
        outputs=[],
    )

    def save() -> None:
        tmp = out / "manifest.json.tmp"
        tmp.write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
        tmp.replace(out / "manifest.json")

    def clean() -> None:
        require(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
            == a.commit,
            "revision differs",
        )
        require(
            not subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip(),
            "dirty checkout",
        )

    def interrupted(signum: int, frame: object) -> None:
        raise RuntimeError(f"signal {signum}")

    save()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        clean()
        unchanged = capture(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--exit-code",
                GATE_COMMIT,
                a.commit,
                "--",
                "src",
                "tests/test_me3j.py",
                "tests/test_mscheme3.py",
                "tests/test_run3.py",
                "tests/test_zero_diagnostic.py",
                "tests/test_zero_oracle.py",
                "tools/comparison/compatibility.py",
            ]
        )
        require(unchanged["returncode"] == 0, "validated producer/tests/checker changed")
        previous = gate(gate_path)
        require(sha(source) == SOURCE_SHA, "source hash")
        m["inputs"] = [
            dict(path=str(source), sha256=SOURCE_SHA),
            dict(path=str(gate_path), sha256=GATE_SHA),
        ]
        os.environ.update({k: "1" for k in THREADS})
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        env.pop("PYTHONOPTIMIZE", None)
        m["threads"] = {k: env[k] for k in THREADS}
        m["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        m["hardware"], m["compiler"] = capture(["lscpu"]), capture(["cc", "--version"])
        import numpy as np

        from nuham_mscheme import run3

        require(Path(run3.__file__).resolve() == ROOT / "src/nuham_mscheme/run3.py", "wrong import")
        require(run3.N_CHUNKS == 256, "chunk count")
        m["params"]["taus"] = list(run3.TAUS)
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        m["blas"] = config.getvalue()
        m["environment_matches_gate"] = all(
            m[k] == previous[k] for k in ("python", "architecture", "blas")
        )
        if not a.smoke_only:
            require(m["environment_matches_gate"], "pilot environment differs from validated gate")
            require(
                m["packages"].get("numpy") == previous["packages"].get("numpy"), "NumPy changed"
            )
        save()

        def sweep(name: str, use: tuple[int, int, int], workers: int) -> dict[str, Any]:
            clean()
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
            stage: dict[str, Any] = dict(name=name, command=cmd, status="running")
            m["stages"].append(stage)
            save()
            tick = time.monotonic()
            with (out / (name + ".log")).open("w") as log:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            stage.update(
                exit_code=proc.returncode,
                wall_time_s=time.monotonic() - tick,
                max_child_rss_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                memory_scope="largest child, not simultaneous aggregate",
                status="passed" if proc.returncode == 0 else "failed",
            )
            save()
            require(proc.returncode == 0, "failed stage " + name)
            doc = json.loads(target.read_text())
            require(doc["git_commit"] == a.commit and doc["git_dirty"] is False, "output revision")
            require(doc["input"] == dict(path=str(source), sha256=SOURCE_SHA), "output source")
            require(doc["file_limits"] == dict(emax=8, e2max=16, e3max=8), "output layout")
            require(
                doc["filters"] == dict(zip(("emax", "e2max", "e3max"), use, strict=True)),
                "output filter",
            )
            require(doc["n_chunks"] == 256 and doc["taus"] == list(run3.TAUS), "output policy")
            invariants(doc)
            return doc

        one = sweep("smoke-w1", (2, 4, 4), 1)
        three = sweep("smoke-w3", (2, 4, 4), 3)
        require(one["parts"] == three["parts"], "worker mismatch")
        reference = json.loads((gate_path.parent / "smoke-w1.json").read_text())
        require(not differences(reference["parts"], one["parts"]), "smoke reference mismatch")
        if not a.smoke_only:
            pilot = sweep("p6", USE, 3)
            m["complement"] = dict(
                status="running", selection="not (bra retained and ket retained)"
            )
            save()
            tick = time.monotonic()
            removed, counts = discarded_stats(source, LIMITS, USE)
            complement: dict[str, Any] = dict(
                selection="not (bra retained and ket retained)",
                file_limits=LIMITS,
                retained_limits=USE,
                **counts,
                parts=parts(removed),
                taus=list(run3.TAUS),
            )
            (out / "complement.json").write_text(json.dumps(complement, indent=2) + "\n")
            require(counts["excluded_tuples"] > 0, "empty complement")
            invariants(complement)
            reference = json.loads((gate_path.parent / "baseline.json").read_text())
            reconstruct(reference["parts"], pilot["parts"], complement["parts"])
            m["complement"] = dict(status="passed", wall_time_s=time.monotonic() - tick, **counts)
        clean()
        require(sha(source) == SOURCE_SHA, "source changed")
        gate(gate_path)
        m.update(status="passed", exit_code=0, worker_results_exact=True)
    except BaseException as exc:
        m.update(status="failed", exit_code=1, error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        m["wall_time_s"] = time.monotonic() - start
        m["self_max_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        m["outputs"] = [
            dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "manifest.json"
        ]
        save()
        print(json.dumps({k: m[k] for k in ("status", "wall_time_s", "exit_code")}))


if __name__ == "__main__":
    main()
