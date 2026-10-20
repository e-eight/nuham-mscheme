"""Exact small-entry oracle for frozen diagnostic fixtures.

Implements the already specified scalar coupling sum in docs/spec.md section 3.
Assumptions: real scalar 3N operator, file-order orbit triples, normalized ordered
determinants, MeV coefficients. CG factors are dimensionless. No terms are
dropped: the stored binary floats become exact rationals before contraction.
This checks the frozen represented input, not an ideal unrounded random operator.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from sympy import Expr, Rational, S, simplify
from sympy.physics.wigner import clebsch_gordan

from nuham_mscheme.me3j import Me3jTuple
from nuham_mscheme.mscheme3 import TripleBasis


def exact_element(
    basis: TripleBasis,
    tuples: Sequence[Me3jTuple],
    bra: Sequence[int],
    ket: Sequence[int],
) -> Expr:
    """Evaluate one element with exact CG factors and exact stored coefficients."""
    if list(bra) != sorted(set(bra)) or list(ket) != sorted(set(ket)):
        raise ValueError("expected canonical unique states")
    if len(bra) != 3 or len(ket) != 3:
        raise ValueError("expected triples")

    def ordered(states: Sequence[int]) -> tuple[list[tuple[int, int, int, int]], int]:
        decoded = sorted([(*basis.decode(g), g) for g in states], key=lambda v: (-v[1], v[3]))
        sign = (-1) ** sum(a[3] > b[3] for a, b in itertools.combinations(decoded, 2))
        return decoded, sign

    b, sb = ordered(bra)
    k, sk = ordered(ket)
    if sum(x[2] for x in b) != sum(x[2] for x in k):
        return S.Zero
    if sum(x[0] for x in b) != sum(x[0] for x in k):
        return S.Zero
    ob, ok = (b[0][1], b[1][1], b[2][1]), (k[0][1], k[1][1], k[2][1])
    if ob < ok:
        b, k, ob, ok = k, b, ok, ob
    table = {(t.bra, t.ket): t for t in tuples}
    if (ob, ok) not in table:
        return S.Zero
    tup = table[(ob, ok)]

    def factor(
        states: list[tuple[int, int, int, int]], j12: int, t12: int, jj: int, tt: int
    ) -> Expr:
        j = [Rational(basis.orbits[x[1]].j2, 2) for x in states]
        m = [Rational(x[2], 2) for x in states]
        t = [Rational(2 * x[0] - 1, 2) for x in states]
        half = Rational(1, 2)
        return (
            clebsch_gordan(j[0], j[1], j12, m[0], m[1], m[0] + m[1])
            * clebsch_gordan(j12, j[2], Rational(jj, 2), m[0] + m[1], m[2], sum(m))
            * clebsch_gordan(half, half, t12, t[0], t[1], t[0] + t[1])
            * clebsch_gordan(t12, half, Rational(tt, 2), t[0] + t[1], t[2], sum(t))
        )

    jb = [basis.orbits[x[1]].j2 for x in b]
    jk = [basis.orbits[x[1]].j2 for x in k]
    terms, i = [], 0
    # Independent channel enumeration; do not call the producer's _slots/tuple_entries.
    for j12 in range(abs(jb[0] - jb[1]) // 2, (jb[0] + jb[1]) // 2 + 1):
        for j45 in range(abs(jk[0] - jk[1]) // 2, (jk[0] + jk[1]) // 2 + 1):
            for jj in range(
                max(abs(2 * j12 - jb[2]), abs(2 * j45 - jk[2])),
                min(2 * j12 + jb[2], 2 * j45 + jk[2]) + 1,
                2,
            ):
                for t12, t45, tt in ((0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1), (1, 1, 3)):
                    value = float(tup.values[i])
                    i += 1
                    if value:
                        coefficient = Rational(*value.as_integer_ratio())
                        terms.append(
                            coefficient * factor(b, j12, t12, jj, tt) * factor(k, j45, t45, jj, tt)
                        )
    if i != len(tup.values):
        raise ValueError("coupled channel coverage differs")
    return simplify(sb * sk * sum(terms, S.Zero))


def exact_stored_contraction(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> Expr:
    """Contract one row/matrix/column exactly as stored, independent of BLAS."""
    if middle.shape != (len(left), len(right)):
        raise ValueError("contraction shapes differ")
    total = S.Zero
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            total += (
                Rational(*float(a).as_integer_ratio())
                * Rational(*float(middle[i, j]).as_integer_ratio())
                * Rational(*float(b).as_integer_ratio())
            )
    return total


def inspect_trace(data: dict, entry: dict, trace: dict, diagnostic: ModuleType) -> dict:
    """Compare the two stored arithmetic paths for one canonical matrix entry."""
    tuples = diagnostic.unpack(data)
    basis = TripleBasis(diagnostic.Layout(*diagnostic.LIMITS).orbits)
    tup = tuples[entry["tuple_id"]]
    gb, gk = tuple(entry["bra"]), tuple(entry["ket"])
    ob = tuple(sorted((basis.decode(g)[1] for g in gb), reverse=True))
    if ob != tup.bra:
        gb, gk = gk, gb
    tt = diagnostic.kernel.tuple_tensor(basis, tup, signed=True)
    for block in diagnostic.kernel.triple_blocks(basis, tup):
        br = [tuple(map(int, g)) for g in block.bra_g]
        kr = [tuple(map(int, g)) for g in block.ket_g]
        if gb not in br or gk not in kr:
            continue
        r, c = br.index(gb), kr.index(gk)
        ab, ak, m2 = block.assign_bra, block.assign_ket, block.m2
        prefix = f"s_{ab}_{ak}_{m2}_"
        p = int(np.flatnonzero((tt.pair_bra == ab) & (tt.pair_ket == ak))[0])
        mi = int(np.flatnonzero(tt.m2 == m2)[0])
        rb = int(np.flatnonzero(tt.ok_bra[p, mi])[r])
        rk = int(np.flatnonzero(tt.ok_ket[p, mi])[c])
        rr, cc = int(trace[prefix + "rows"][r]), int(trace[prefix + "cols"][c])
        rows, scalar_exact, batch_exact = [], S.Zero, S.Zero
        sign = int(block.bra_sign[r] * block.ket_sign[c])
        for y, jj in enumerate(trace["js"]):
            if jj < abs(m2):
                continue
            left = trace["scalar_spb"][rr, :, y]
            right = trace["scalar_spk"][cc, :, y]
            middle = trace[prefix + "u"][:, :, y]
            bleft = trace["batch_a"][y, mi, rb]
            bright = trace["batch_bt"][y, mi, :, rk]
            bmiddle = trace["batch_u"][p, y]
            scalar_exact += exact_stored_contraction(left, middle, right)
            batch_exact += exact_stored_contraction(bleft, bmiddle, bright)
            rows.append(
                dict(
                    two_J=int(jj),
                    left_equal=np.array_equal(left, bleft),
                    middle_equal=np.array_equal(middle, bmiddle),
                    right_equal=np.array_equal(right, bright),
                    first_product_equal=np.array_equal(
                        trace[prefix + f"first_{y}"][r], trace[f"batch_first_{y}"][p, mi, rb]
                    ),
                    scalar_term=float(trace[prefix + f"term_{y}"][r, c]),
                    batch_term=float(trace[f"batch_term_{y}"][p, mi, rb, rk]),
                    scalar_partial=float(trace[prefix + f"sum_{y}"][r, c]),
                    batch_partial=float(trace[f"batch_sum_{y}"][p, mi, rb, rk]),
                )
            )
        return dict(
            sign=sign,
            contractions=rows,
            scalar_stored_exact=str(sign * scalar_exact),
            batch_stored_exact=str(sign * batch_exact),
            scalar_stored_decimal=str((sign * scalar_exact).evalf(80)),
            batch_stored_decimal=str((sign * batch_exact).evalf(80)),
            equal_exact_stored=bool(scalar_exact == batch_exact),
        )
    raise ValueError("entry not found in tuple")


def main() -> None:
    """Analyze retrieved evidence locally, with bounded separate exact-oracle processes."""
    import diagnose_zeros3 as diagnostic

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--local-run", type=Path, help="local native + imported ARM fixture replay")
    ap.add_argument("--entry-fixture", type=Path)
    ap.add_argument("--bra")
    ap.add_argument("--ket")
    args = ap.parse_args()
    if args.entry_fixture:
        tuples = diagnostic.unpack(diagnostic.load(args.entry_fixture))
        basis = TripleBasis(diagnostic.Layout(*diagnostic.LIMITS).orbits)
        x = exact_element(
            basis, tuples, list(map(int, args.bra.split(","))), list(map(int, args.ket.split(",")))
        )
        print(
            json.dumps(
                dict(
                    expression=str(x),
                    zero=x.is_zero,  # None stays unresolved; structural inequality is not proof.
                    decimal80=str(x.evalf(80)),
                    decimal160=str(x.evalf(160)),
                )
            )
        )
        return
    if not args.run or not args.out:
        ap.error("--run and --out required")
    args.out.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    fixture = args.run / "native-fixture.npz"
    parent = json.loads((args.run / "manifest.json").read_text())
    diagnostic.require(parent["status"] == "diagnostic_complete", "parent incomplete")
    for f in parent["outputs"]:
        diagnostic.require(
            diagnostic.sha(args.run / f["path"]) == f["sha256"], "parent artifact hash"
        )
    manifest: dict[str, Any] = dict(
        status="running",
        started_utc=time.strftime("%FT%TZ", time.gmtime()),
        git_commit=diagnostic.git("rev-parse", "HEAD"),
        git_status=diagnostic.git("status", "--porcelain"),
        python=sys.version,
        numpy=np.__version__,
        host=diagnostic.platform.node(),
        architecture=diagnostic.platform.machine(),
        packages={
            d.metadata["Name"]: d.version for d in diagnostic.importlib.metadata.distributions()
        },
        threads={k: diagnostic.os.environ.get(k) for k in diagnostic.THREADS},
        parent_commit=parent["git_commit"],
        inputs=[
            dict(path=str(fixture), sha256=diagnostic.sha(fixture)),
            dict(
                path=str(args.run / "manifest.json"),
                sha256=diagnostic.sha(args.run / "manifest.json"),
            ),
        ],
        script_sha256=diagnostic.sha(Path(__file__)),
        per_entry_timeout_s=30,
    )
    diagnostic.dump(args.out / "manifest.json", manifest)
    try:
        diagnostic.require(not manifest["git_status"], "analysis requires clean checkout")
        if args.local_run:
            local_manifest = json.loads((args.local_run / "manifest.json").read_text())
            diagnostic.require(
                local_manifest["status"] == "diagnostic_complete", "local incomplete"
            )
            for f in local_manifest["outputs"]:
                diagnostic.require(
                    diagnostic.sha(args.local_run / f["path"]) == f["sha256"], "local artifact hash"
                )
            manifest["inputs"].append(
                dict(
                    path=str(args.local_run / "manifest.json"),
                    sha256=diagnostic.sha(args.local_run / "manifest.json"),
                )
            )
            diagnostic.require(
                parent["native_fingerprints"] == local_manifest["frozen_fingerprints"],
                "ARM fixture identity",
            )
            diagnostic.require(
                parent["frozen_fingerprints"] == local_manifest["native_fingerprints"],
                "laptop fixture identity",
            )
            diagnostic.dump(args.out / "manifest.json", manifest)
            comparisons = {}
            for origin, arm, local in (
                ("laptop", args.run / "frozen", args.local_run / "native"),
                ("ARM", args.run / "native", args.local_run / "frozen"),
            ):
                for path in ("dense", "batched"):
                    with np.load(arm / (path + ".npz")) as a, np.load(local / (path + ".npz")) as b:
                        diagnostic.require(a.files == b.files, "block coverage differs")
                        keys = [k for k in a.files if not k.endswith("_states")]
                        for k in a.files:
                            if k.endswith("_states"):
                                diagnostic.require(np.array_equal(a[k], b[k]), "state ordering")
                        comparisons[origin + "/" + path] = dict(
                            zero_mask_differences=sum(
                                int(np.count_nonzero((a[k] == 0) != (b[k] == 0))) for k in keys
                            ),
                            different_values=sum(int(np.count_nonzero(a[k] != b[k])) for k in keys),
                            max_abs_difference=max(float(np.abs(a[k] - b[k]).max()) for k in keys),
                            bitwise_equal=all(a[k].tobytes() == b[k].tobytes() for k in keys),
                        )
            diagnostic.dump(args.out / "cross-platform.json", comparisons)
        entries = json.loads((args.run / "native/mismatches.json").read_text())
        data = diagnostic.load(fixture)
        tuples = diagnostic.unpack(data)
        basis = TripleBasis(diagnostic.Layout(*diagnostic.LIMITS).orbits)
        result = []
        for tid in sorted({e["tuple_id"] for e in entries}):
            diagnostic.trace_tuple(basis, tuples[tid], args.out / f"local-trace-{tid}.npz")
        for entry in entries:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--entry-fixture",
                str(fixture),
                "--bra",
                ",".join(map(str, entry["bra"])),
                "--ket",
                ",".join(map(str, entry["ket"])),
            ]
            record = dict(entry=entry, command=cmd)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
                record["exact_couplings"] = json.loads(proc.stdout)
            except subprocess.TimeoutExpired:
                record["exact_couplings"] = {"status": "unresolved_timeout"}
            for label, p in (
                ("ARM", args.run / "native" / f"trace-{entry['tuple_id']}.npz"),
                ("local", args.out / f"local-trace-{entry['tuple_id']}.npz"),
            ):
                if p.exists():
                    with np.load(p, allow_pickle=False) as z:
                        trace = {k: z[k] for k in z.files}
                    record[label] = inspect_trace(data, entry, trace, diagnostic)
            result.append(record)
            diagnostic.dump(args.out / "entries.json", result)
        manifest.update(status="complete", exit_code=0, entries=len(result))
    except BaseException as exc:
        manifest.update(status="failed", exit_code=1, error=repr(exc))
        raise
    finally:
        manifest["wall_time_s"] = time.monotonic() - tick
        manifest["outputs"] = [
            dict(path=str(p.relative_to(args.out)), sha256=diagnostic.sha(p))
            for p in args.out.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        ]
        diagnostic.dump(args.out / "manifest.json", manifest)
        print(json.dumps({k: manifest[k] for k in ("status", "wall_time_s")}))


if __name__ == "__main__":
    main()
