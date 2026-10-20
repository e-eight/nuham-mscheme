"""Freeze and compare the two existing 3N evaluation paths at toy size.

No producer kernels or test assertions are changed. A completed diagnosis may
contain zero-pattern differences; it is not a passed compatibility gate.
"""

# Imports below deliberately select this checkout rather than an installed producer.
# ruff: noqa: E402
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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np

original = __import__("test_mscheme3")

from nuham_mscheme import mscheme3 as kernel
from nuham_mscheme.dense3 import dense_blocks
from nuham_mscheme.me3j import Layout, Me3jTuple, kept_intervals
from nuham_mscheme.stats2 import TAUS
from nuham_mscheme.stats3 import ThreeBodyStats

LIMITS = (2, 4, 4)
BASE = "c69f815c15411b4a1ac16802aff6e2d99967baa5"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def require(ok: bool, message: str) -> None:
    """Reject a failed diagnostic prerequisite without relying on assert."""
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash a file in bounded chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def git(*args: str) -> str:
    """Read metadata from this checkout."""
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def dump(path: Path, value: object) -> None:
    """Write JSON atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def raw_values(lay: Layout, bra: int, ket: int, rng: np.random.Generator) -> np.ndarray:
    """Repeat the original fixture arithmetic, retaining the pre-cleanup values."""
    jb, jk = lay.jkeys[bra], lay.jkeys[ket]
    j12s, j45s, js, index = kernel._slots(jb, jk)
    vt = np.zeros((len(j12s), 2, len(j45s), 2, len(js), 2))
    vt[tuple(index.T)] = rng.normal(size=len(index))
    if bra == ket:
        vt = 0.5 * (vt + vt.transpose(2, 3, 0, 1, 4, 5))
    pib = original._antisymmetrizer(lay.triples[bra], jb, j12s, js)
    pik = original._antisymmetrizer(lay.triples[ket], jk, j45s, js)
    step = np.einsum("yTxtXs,XskuyT->xtkuyT", pib, vt)
    vt = np.einsum("xtKUyT,yTKUku->xtkuyT", step, pik)
    return vt[tuple(index.T)]


def native_fixture() -> dict[str, np.ndarray]:
    """Generate and verify bitwise equality with the original seeded fixture."""
    lay = Layout(*LIMITS)
    rng = np.random.default_rng(20261010)
    keys, values, before, offsets = [], [], [], [0]
    for _, _, i, k in kept_intervals(lay, np.ones(len(lay.triples), dtype=bool)):
        raw = raw_values(lay, i, k, rng)
        v = raw.copy()
        v[np.abs(v) < 1e-12] = 0.0
        keys.append((*lay.triples[i], *lay.triples[k]))
        before.append(raw)
        values.append(v)
        offsets.append(offsets[-1] + len(v))
    data = {
        "limits": np.array(LIMITS, dtype="<i8"),
        "keys": np.array(keys, dtype="<i8"),
        "offsets": np.array(offsets, dtype="<i8"),
        "values": np.concatenate(values).astype("<f8"),
        "before_cleanup": np.concatenate(before).astype("<f8"),
    }
    frozen = unpack(data)
    reference = original._synthetic()
    require(len(frozen) == len(reference), "fixture tuple count")
    for a, b in zip(frozen, reference, strict=True):
        require(a.bra == b.bra and a.ket == b.ket, "fixture ordering")
        require(a.values.tobytes() == b.values.tobytes(), "fixture arithmetic changed")
    return data


def unpack(data: dict[str, np.ndarray]) -> list[Me3jTuple]:
    """Validate and decode the small lossless fixture format."""
    require(
        set(data) == {"limits", "keys", "offsets", "values", "before_cleanup"}, "fixture fields"
    )
    require(tuple(data["limits"]) == LIMITS, "fixture limits")
    keys, offsets, values = data["keys"], data["offsets"], data["values"]
    require(keys.dtype == np.dtype("<i8") and keys.ndim == 2 and keys.shape[1] == 6, "keys")
    require(offsets.dtype == np.dtype("<i8") and len(offsets) == len(keys) + 1, "offsets")
    require(offsets[0] == 0 and offsets[-1] == len(values), "offset endpoints")
    require(bool(np.all(np.diff(offsets) > 0)), "offset order")
    require(values.dtype == np.dtype("<f8") and values.ndim == 1, "values")
    require(data["before_cleanup"].dtype == values.dtype, "raw dtype")
    require(data["before_cleanup"].shape == values.shape, "raw shape")
    require(bool(np.isfinite(values).all() and np.isfinite(data["before_cleanup"]).all()), "finite")
    lay = Layout(*LIMITS)
    expected = [
        (*lay.triples[i], *lay.triples[k])
        for _, _, i, k in kept_intervals(lay, np.ones(len(lay.triples), dtype=bool))
    ]
    require(keys.tolist() == [list(k) for k in expected], "canonical key coverage")
    return [
        Me3jTuple(
            (int(k[0]), int(k[1]), int(k[2])),
            (int(k[3]), int(k[4]), int(k[5])),
            values[offsets[i] : offsets[i + 1]],
        )
        for i, k in enumerate(keys)
    ]


def load(path: Path) -> dict[str, np.ndarray]:
    """Load a fixture without pickle."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    unpack(data)
    return data


def fingerprints(data: dict[str, np.ndarray]) -> dict[str, str]:
    """Hash canonical array bytes independently of NPZ container metadata."""
    return {k: hashlib.sha256(v.tobytes(order="C")).hexdigest() for k, v in data.items()}


def describe(array: np.ndarray) -> dict[str, Any]:
    """Preserve in-memory layout information lost by array serialization."""
    return dict(shape=list(array.shape), strides=list(array.strides), dtype=str(array.dtype))


def trace_tuple(basis: kernel.TripleBasis, tup: Me3jTuple, path: Path) -> None:
    """Capture both contraction paths without altering their matrix shapes."""
    jb = tuple(basis.orbits[i].j2 for i in tup.bra)
    jk = tuple(basis.orbits[i].j2 for i in tup.ket)
    j12s, j45s, js, index = kernel._slots(jb, jk)
    vt = np.zeros((len(j12s), 2, len(j45s), 2, len(js), 2))
    vt[tuple(index.T)] = tup.values
    pb, pk, _ = kernel._pairs()
    iso, _ = kernel._isospin()
    u = np.einsum("pxT,pyT,ixkyJT->pJik", iso[pb], iso[pk], vt)
    rb, rk = basis.triple_rows(tup.bra), basis.triple_rows(tup.ket)
    mc = min(rb.m_max, rk.m_max)
    ob, ok = (rb.m_max - mc) // 2, (rk.m_max - mc) // 2
    rows_b, rows_k = rb.rows[ob : ob + mc + 1], rk.rows[ok : ok + mc + 1]
    js_b, sp_b = kernel._class_spatial(jb)
    js_k, sp_k = kernel._class_spatial(jk)
    a = kernel._take_j(js_b, sp_b, js)[:, rows_b]
    bt = kernel._take_j(js_k, sp_k, js)[:, rows_k].transpose(0, 1, 3, 2)
    arrays = dict(
        vt=vt,
        iso=iso,
        js=np.array(js),
        slot_index=index,
        batch_u=u,
        batch_a=a,
        batch_bt=bt,
        rows_b=rows_b,
        rows_k=rows_k,
    )
    meta: dict[str, Any] = {k: describe(v) for k, v in arrays.items()}
    x = np.zeros((len(pb), mc + 1, rows_b.shape[1], rows_k.shape[1]))
    for y in range(len(js)):
        first = a[y][None] @ u[:, y, None]
        term = first @ bt[y][None]
        x += term
        arrays[f"batch_first_{y}"] = first
        arrays[f"batch_term_{y}"] = term
        arrays[f"batch_sum_{y}"] = x.copy()
    expected = kernel.tuple_tensor(basis, tup)
    meta["batch_replay_equal"] = x.tobytes() == expected.x.tobytes()
    arrays["batch_original"] = expected.x
    arrays["batch_valid"] = expected.valid
    spb = kernel._spatial(jb, j12s, js).reshape(-1, len(j12s), len(js))
    spk = kernel._spatial(jk, j45s, js).reshape(-1, len(j45s), len(js))
    arrays.update(scalar_spb=spb, scalar_spk=spk)
    for block in kernel.triple_blocks(basis, tup):
        ab, ak, m2 = block.assign_bra, block.assign_ket, block.m2
        prefix = f"s_{ab}_{ak}_{m2}_"
        states_b, states_k = basis.slot_states(tup.bra, ab), basis.slot_states(tup.ket, ak)
        rows = np.flatnonzero(states_b.ok & (states_b.m2 == m2))
        cols = np.flatnonzero(states_k.ok & (states_k.m2 == m2))
        scalar_u = np.einsum("xT,yT,ixkyJT->ikJ", iso[ab], iso[ak], vt)
        arrays[prefix + "u"] = scalar_u
        arrays[prefix + "rows"] = rows
        arrays[prefix + "cols"] = cols
        scalar = np.zeros((len(rows), len(cols)))
        for y, jj in enumerate(js):
            if jj >= abs(m2):
                left, right = spb[rows, :, y], spk[cols, :, y].T
                first = left @ scalar_u[:, :, y]
                term = first @ right
                scalar += term
                arrays[prefix + f"first_{y}"] = first
                arrays[prefix + f"term_{y}"] = term
                arrays[prefix + f"sum_{y}"] = scalar.copy()
                meta[prefix + f"left_{y}"] = describe(left)
                meta[prefix + f"right_{y}"] = describe(right)
        scalar *= states_b.sign[rows][:, None] * states_k.sign[cols][None, :]
        meta[prefix + "replay_equal"] = scalar.tobytes() == block.x.tobytes()
        arrays[prefix + "original"] = block.x
    meta.update({k: describe(v) for k, v in arrays.items() if k not in meta})
    np.savez_compressed(path, allow_pickle=False, **arrays)
    dump(path.with_suffix(".json"), meta)


def evaluate(data: dict[str, np.ndarray], out: Path) -> dict[str, Any]:
    """Compare all toy entries and separately validate counting of batched entries."""
    out.mkdir()
    tuples = unpack(data)
    basis = kernel.TripleBasis(Layout(*LIMITS).orbits)
    dense = dense_blocks(basis, tuples)
    pos = {k: {g: i for i, g in enumerate(states)} for k, (_, states) in dense.items()}
    batch = {k: np.zeros_like(mat) for k, (mat, _) in dense.items()}
    owner = {k: np.full(mat.shape, -1, dtype=np.int32) for k, (mat, _) in dense.items()}
    stats = ThreeBodyStats(basis)
    for tid, tup in enumerate(tuples):
        stats.add_tuple(tup)
        tt = kernel.tuple_tensor(basis, tup, signed=True)
        for p, (ab, ak) in enumerate(zip(tt.pair_bra, tt.pair_ket, strict=True)):
            for mi, m2 in enumerate(tt.m2):
                rb = np.flatnonzero(tt.ok_bra[p, mi])
                rk = np.flatnonzero(tt.ok_ket[p, mi])
                if not len(rb) or not len(rk):
                    continue
                key = (int(tt.pair_sector[p]), int(m2))
                gb = basis.slot_states(tup.bra, int(ab)).g[tt.rows_bra[mi, rb]]
                gk = basis.slot_states(tup.ket, int(ak)).g[tt.rows_ket[mi, rk]]
                rows = [pos[key][(int(g[0]), int(g[1]), int(g[2]))] for g in gb]
                cols = [pos[key][(int(g[0]), int(g[1]), int(g[2]))] for g in gk]
                values = tt.x[p, mi][np.ix_(rb, rk)]
                writes = [(rows, cols, values)]
                if tup.bra != tup.ket:
                    writes.append((cols, rows, values.T))
                for rr, cc, xx in writes:
                    ix = np.ix_(rr, cc)
                    require(bool(np.all(owner[key][ix] == -1)), "duplicate batched write")
                    batch[key][ix] = xx
                    owner[key][ix] = tid
    summaries, mismatches, dense_archive, batch_archive = {}, [], {}, {}
    blocks = stats.blocks()
    for key, (d, states) in dense.items():
        s, m2 = key
        name = f"{kernel.SECTORS3[s]}:{m2}"
        b = batch[key]
        st = blocks[(kernel.SECTORS3[s], m2)]
        direct = []
        for tau in TAUS:
            for label, mask, expected in (
                ("full", np.abs(b) > tau, st.nnz_full[TAUS.index(tau)]),
                ("ut", np.triu(np.abs(b)) > tau, st.nnz_ut(TAUS.index(tau))),
                ("diag", np.abs(np.diag(b)) > tau, st.nnz_diag[TAUS.index(tau)]),
            ):
                have = int(np.count_nonzero(mask))
                require(have == expected, f"accumulator/direct {name}/{label}/{tau}")
            direct.append(int(np.count_nonzero(np.abs(b) > tau)))
        require(int(np.count_nonzero(owner[key] >= 0)) == st.structural_full, "structural mask")
        for have, expected in (
            (math.fsum(np.abs(b).flat), st.lambda_full),
            (math.fsum(np.abs(np.diag(b))), st.lambda_diag),
        ):
            require(math.isclose(have, expected, rel_tol=1e-12, abs_tol=1e-12), "direct norm")
        zero_mask = (d == 0) != (b == 0)
        summaries[name] = dict(
            dense_nnz=[int(np.count_nonzero(np.abs(d) > t)) for t in TAUS],
            batched_nnz=direct,
            zero_disagreements=int(zero_mask.sum()),
            max_abs_difference=float(np.abs(d - b).max()),
            dense_norm=math.fsum(np.abs(d).flat),
            batch_norm=st.lambda_full,
            batch_only_nonzero=int(np.count_nonzero((d == 0) & (b != 0))),
            dense_only_nonzero=int(np.count_nonzero((d != 0) & (b == 0))),
        )
        for r, c in np.argwhere(zero_mask):
            tid = int(owner[key][r, c])
            require(tid >= 0, "nonzero outside written support")
            tup = tuples[tid]
            mismatches.append(
                dict(
                    block=name,
                    row=int(r),
                    col=int(c),
                    bra=list(states[r]),
                    ket=list(states[c]),
                    tuple_id=tid,
                    tuple_bra=list(tup.bra),
                    tuple_ket=list(tup.ket),
                    dense=float(d[r, c]),
                    batched=float(b[r, c]),
                    dense_hex=float(d[r, c]).hex(),
                    batched_hex=float(b[r, c]).hex(),
                    mirror_dense=float(d[c, r]),
                    mirror_batched=float(b[c, r]),
                    mirror_disagrees=bool(zero_mask[c, r]),
                )
            )
        dense_archive[name], batch_archive[name] = d, b
        dense_archive[name + "_states"] = np.array(states, dtype="<i8")
    np.savez_compressed(out / "dense.npz", **dense_archive)
    np.savez_compressed(out / "batched.npz", **batch_archive)
    dump(out / "mismatches.json", mismatches)
    dump(out / "blocks.json", summaries)
    selected = sorted({e["tuple_id"] for e in mismatches})
    # Save every implicated tuple; trace at most eight distinct cases to bound job cost.
    for tid in selected[:8]:
        trace_tuple(basis, tuples[tid], out / f"trace-{tid}.npz")
    dump(
        out / "trace-selection.json",
        dict(
            all_implicated=selected,
            traced=selected[:8],
            untraced=selected[8:],
            reason="bounded diagnostic",
        ),
    )
    return dict(
        zero_disagreements=len(mismatches),
        implicated_tuples=selected,
        direct_count_checks="all blocks, full/upper/diagonal, all thresholds passed",
    )


def main() -> None:
    """Run a bounded native and optional frozen-input comparison."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--allow-dirty", action="store_true", help="local development only")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    m: dict[str, Any] = dict(
        status="running",
        started_utc=time.strftime("%FT%TZ", time.gmtime()),
        git_commit=git("rev-parse", "HEAD"),
        git_status=git("status", "--porcelain"),
        git_diff=git("diff", "HEAD"),
        host=platform.node(),
        architecture=platform.machine(),
        python=sys.version,
        seeds={"synthetic": 20261010},
        limits=LIMITS,
        taus=list(TAUS),
        env={k: os.environ.get(k) for k in THREADS},
        slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        inputs=[],
        outputs=[],
    )
    dump(out / "manifest.json", m)
    try:
        require(not m["git_status"] or args.allow_dirty, "dirty diagnostic checkout")
        require(all(os.environ.get(k) == "1" for k in THREADS), "set all BLAS threads to one")
        # These must remain byte-identical to the failed producer, including the original tests.
        paths = git("ls-tree", "-r", "--name-only", BASE, "src", "tests").splitlines()
        for rel in paths:
            original_bytes = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", BASE + ":" + rel]
            )
            require((ROOT / rel).read_bytes() == original_bytes, "changed producer file: " + rel)
        m["producer_hashes"] = {p: sha(ROOT / p) for p in paths}
        m["diagnostic_sha256"] = sha(Path(__file__))
        m["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        m["numpy_config"] = config.getvalue()
        m["cpu"] = subprocess.check_output(["lscpu"], text=True)
        m["compiler"] = subprocess.run(["cc", "--version"], capture_output=True, text=True).stdout
        if args.fixture:
            m["inputs"].append(dict(path=str(args.fixture.resolve()), sha256=sha(args.fixture)))
        dump(out / "manifest.json", m)
        native = native_fixture()
        np.savez_compressed(out / "native-fixture.npz", allow_pickle=False, **native)
        require(
            fingerprints(load(out / "native-fixture.npz")) == fingerprints(native), "round trip"
        )
        m["native_fingerprints"] = fingerprints(native)
        dump(out / "manifest.json", m)
        m["native_result"] = evaluate(native, out / "native")
        dump(out / "manifest.json", m)
        if args.fixture:
            frozen = load(args.fixture)
            require(np.array_equal(frozen["keys"], native["keys"]), "cross-fixture keys")
            require(np.array_equal(frozen["offsets"], native["offsets"]), "cross-fixture offsets")
            m["frozen_fingerprints"] = fingerprints(frozen)
            m["fixture_comparison"] = {
                k: dict(
                    different_values=int(np.count_nonzero(native[k] != frozen[k])),
                    zero_mask_disagreements=int(
                        np.count_nonzero((native[k] == 0) != (frozen[k] == 0))
                    ),
                    max_abs_difference=float(np.max(np.abs(native[k] - frozen[k]))),
                )
                for k in ("values", "before_cleanup")
            }
            m["frozen_result"] = evaluate(frozen, out / "frozen")
        require(git("status", "--porcelain") == m["git_status"], "checkout changed")
        require(git("rev-parse", "HEAD") == m["git_commit"], "revision changed")
        require(
            all(sha(ROOT / p) == h for p, h in m["producer_hashes"].items()), "producer changed"
        )
        require(all(sha(Path(f["path"])) == f["sha256"] for f in m["inputs"]), "input changed")
        m.update(
            status="diagnostic_complete",
            exit_code=0,
            interpretation="Differences are evidence; this is not a compatibility pass.",
        )
    except BaseException as exc:
        m.update(status="failed", exit_code=1, error=repr(exc))
        raise
    finally:
        m["wall_time_s"] = time.monotonic() - start
        m["max_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        m["outputs"] = [
            dict(path=str(p.relative_to(out)), sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        ]
        dump(out / "manifest.json", m)
        print(json.dumps({k: m[k] for k in ("status", "wall_time_s", "max_rss_KiB")}))


if __name__ == "__main__":
    main()
