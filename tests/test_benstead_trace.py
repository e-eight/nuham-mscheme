"""Acceptance checks 1-5 of ``docs/benstead-block-trace-spec.md`` on the emax 4 fixture."""

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
E4_NAME = "TwBME-HO_NN-only_N3LO_EM500_srg2.0_hw16_emax4_e2max8.me2j.gz"
E4_CANDIDATES = (
    ROOT.parent / "nuhamil-alt" / "refs" / "work" / E4_NAME,
    ROOT.parent / "nuham-jax" / "refs" / "work" / E4_NAME,
)
WORK = Path("/tmp/nuham-imsrg")

sys.path.insert(0, str(ROOT / "tools"))
import benstead_trace as bt  # noqa: E402  # type: ignore

from nuham_mscheme.cg import cg_table  # noqa: E402
from nuham_mscheme.dense2 import sp_states  # noqa: E402
from nuham_mscheme.me2j import iter_tuples  # noqa: E402
from nuham_mscheme.mscheme2 import PN, TInt, quad_blocks  # noqa: E402
from nuham_mscheme.orbits import orbit_list  # noqa: E402
from nuham_mscheme.stats2 import TwoBodyStats  # noqa: E402

KEYS = (
    bt.BlockKey("pp", 16),
    bt.BlockKey("pp", -16),
    bt.BlockKey("pn", 18),
    bt.BlockKey("pn", -18),
)
TINT = TInt(16.0, 16)


@pytest.fixture(scope="session")
def e4_file():
    """The original NuHamil emax 4 file; fall back to the sibling nuham-jax checkout."""
    for candidate in E4_CANDIDATES:
        if candidate.exists():
            return candidate
    pytest.skip(f"reference file missing: {E4_CANDIDATES[0]}")


@pytest.fixture(scope="module")
def fixture_data(e4_file):
    """Build the four fixture blocks and the matching streaming statistics once."""
    build = bt.build_traces(e4_file, 4, 8, KEYS, TINT)
    summaries = {key: bt.summarize(build.traces[key]) for key in KEYS}
    stats = TwoBodyStats(TINT)
    orbits = orbit_list(4)
    for tup in iter_tuples(e4_file, 4, 8):
        stats.add_tuple(orbits, tup)
    report = bt.OracleReport(
        status="not_requested",
        reason=None,
        image="imsrg-dev",
        work_dir=str(WORK),
        modes={"v": "file", "tint": "trel", "v_plus_tint": "file+trel"},
        tolerances=bt.ORACLE_TOL,
    )
    payloads = {
        key: bt.trace_payload(
            build.traces[key],
            str(e4_file),
            bt.sha256_file(e4_file),
            4,
            8,
            TINT,
            report,
        )
        for key in KEYS
    }
    return build, summaries, stats, payloads


def _rel_close(value, expected):
    """Relative comparison at the acceptance tolerance."""
    scale = max(abs(value), abs(expected))
    return abs(value - expected) <= 1e-12 * scale if scale else True


def test_check1_summary_matches_two_body_stats(fixture_data):
    """Check 1: exported summary equals TwoBodyStats on the same file."""
    _, summaries, stats, _ = fixture_data
    for key in KEYS:
        summary = summaries[key]
        for part, stored_part in bt.STORED_PARTS.items():
            got = summary["parts"][part]
            expected = stats.blocks(stored_part)[(key.sector, key.two_m)]
            assert summary["structural_ut"] == expected.structural_ut()
            assert got["nnz_ut_tau0"] == expected.nnz_ut(0)
            assert got["nnz_ut_tau5e-9"] == expected.nnz_ut(1)
            assert got["nnz_diag_tau0"] == expected.nnz_diag[0]
            assert got["nnz_diag_tau5e-9"] == expected.nnz_diag[1]
            assert _rel_close(got["lambda_ut"], expected.lambda_ut)
            assert _rel_close(got["lambda_full"], expected.lambda_full)
            assert _rel_close(got["lambda_diag"], expected.lambda_diag)
            assert _rel_close(got["max_abs"], expected.vmax)


def test_check2_contributions_add_up(fixture_data):
    """Check 2: every V element is the sum of its exported record contributions."""
    _, _, _, payloads = fixture_data
    worst = 0.0
    for payload in payloads.values():
        for element in payload["elements"]:
            records = element["v"]["contributions"]
            total = math.fsum(rec["contribution"] for rec in records)
            residual = abs(total - element["v"]["value"])
            worst = max(worst, residual)
            assert residual <= 1e-13, (payload["block"], element["bra"], element["ket"], residual)
    assert worst >= 0.0


def test_check3_invariants(fixture_data):
    """Check 3: Hermiticity, time reversal and parity on the kernel matrices."""
    build, _, _, _ = fixture_data
    for key in KEYS:
        trace = build.traces[key]
        for part, mat in trace.values.items():
            assert np.array_equal(mat, mat.T), (key, part, np.abs(mat - mat.T).max())
            parities = np.asarray(trace.parities)
            cross = parities[:, None] != parities[None, :]
            assert np.all(mat[cross] == 0.0), (key, part, np.abs(mat[cross]).max())

    for left, right in (
        (bt.BlockKey("pp", 16), bt.BlockKey("pp", -16)),
        (bt.BlockKey("pn", 18), bt.BlockKey("pn", -18)),
    ):
        a, b = build.traces[left], build.traces[right]
        assert len(a.pair_states) == len(b.pair_states)
        for part in ("v", "tint", "v_plus_tint"):
            ua = np.sort(np.abs(np.asarray(bt._ut_values(a.values[part]))))
            ub = np.sort(np.abs(np.asarray(bt._ut_values(b.values[part]))))
            assert np.array_equal(ua, ub), (left, right, part, np.abs(ua - ub).max())


def test_check5_pn_stretched_positive_control(fixture_data, e4_file):
    """Check 5: the stretched pn V equals the kernel value and T_int,2 is zero."""
    build, _, _, _ = fixture_data
    trace = build.traces[bt.BlockKey("pn", 18)]
    assert len(trace.pair_states) == 1
    assert trace.values["tint"][0, 0] == 0.0

    orbits = orbit_list(4)
    sps = sp_states(orbits, 4)
    sindex = {(species, orbit, m2): i for i, (species, orbit, m2) in enumerate(sps)}
    target = trace.pair_states[0]
    expected = 0.0
    for tup in iter_tuples(e4_file, 4, 8):
        oa, ob, oc, od = (orbits[k] for k in (tup.a, tup.b, tup.c, tup.d))
        for blk in quad_blocks(orbits, tup):
            if blk.sector != PN:
                continue
            tb = (0, 1) if blk.bra_assign == 0 else (1, 0)
            tk = (0, 1) if blk.ket_assign == 0 else (1, 0)
            keep = (
                (blk.mb2[:, :, None, None] == blk.mk2[None, None])
                & blk.valid_b[:, :, None, None]
                & blk.valid_k[None, None]
                & (blk.mb2[:, :, None, None] == 18)
            )
            for ia, ib, ic, id_ in np.argwhere(keep):
                ia_i, ib_i, ic_i, id_i = int(ia), int(ib), int(ic), int(id_)
                sa = sindex[(tb[0], tup.a, 2 * ia_i - oa.j2)]
                sb = sindex[(tb[1], tup.b, 2 * ib_i - ob.j2)]
                sc = sindex[(tk[0], tup.c, 2 * ic_i - oc.j2)]
                sd = sindex[(tk[1], tup.d, 2 * id_i - od.j2)]
                bra_pair = (sa, sb) if tb[0] == 0 else (sb, sa)
                ket_pair = (sc, sd) if tk[0] == 0 else (sd, sc)
                if bra_pair != target or ket_pair != target:
                    continue
                sign_a = 1 if sa < sb else -1
                sign_b = 1 if sc < sd else -1
                for jj, jval in enumerate(range(tup.jmin, tup.jmin + len(tup.values))):
                    if blk.bra_assign == blk.ket_assign:
                        weight = 0.5 * (tup.values[jj, 2] + tup.values[jj, 0])
                    else:
                        weight = 0.5 * (tup.values[jj, 2] - tup.values[jj, 0])
                    cg_a = cg_table(oa.j2, ob.j2, 2 * jval)[ia_i, ib_i]
                    cg_b = cg_table(oc.j2, od.j2, 2 * jval)[ic_i, id_i]
                    expected += float(sign_a * sign_b * weight * cg_a * cg_b)
    assert trace.values["v"][0, 0] == pytest.approx(expected, abs=1e-12)
    assert abs(trace.values["v"][0, 0]) > 1.0


@pytest.mark.skipif(
    shutil.which("podman") is None or not (WORK / "imsrg" / "build" / "libIMSRG.so").exists(),
    reason="needs podman and the imsrg++ build",
)
def test_check4_oracle(e4_file):
    """Check 4: every exported element agrees with imsrg++ within the spec tolerances."""
    build = bt.build_traces(e4_file, 4, 8, KEYS, TINT)
    report = bt.run_oracle(e4_file, 4, 8, build.traces, KEYS, TINT)
    assert report.status == "run"
    assert report.violations == []
    for key in KEYS:
        trace = build.traces[key]
        slots = [(i, j) for i in range(len(trace.parities)) for j in range(i, len(trace.parities))]
        for part, tolerance in bt.ORACLE_TOL.items():
            for pos, (i, j) in enumerate(slots):
                diff = abs(float(trace.values[part][i, j]) - trace.oracle[part][pos])
                assert diff <= tolerance, (key, part, i, j, diff)
    assert report.max_abs_diff["v"] > 0.0
