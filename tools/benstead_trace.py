"""Export a tiny two-body m-scheme block with full per-record provenance.

This is the diagnostic tool of ``docs/benstead-block-trace-spec.md``.  For each requested
``(sector, 2M)`` block it writes the unique pair states, every upper-triangle slot with the
file's V, the two-body part of T_int, their sum, the per-record contributions, a summary
computed only from the exported values, an optional imsrg++ oracle comparison, and a
manifest.  The production kernel ``mscheme2.quad_blocks`` is used read-only; nothing in
``src/`` is changed.

Usage::

    python tools/benstead_trace.py [--blocks pn:34,pn:-34,...] [--oracle]

Defaults are the production input of the spec and the blocks pn +-34, pp +-32, nn +-32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nuham_mscheme.cg import cg_table
from nuham_mscheme.dense2 import sp_states
from nuham_mscheme.me2j import Me2jTuple, iter_tuples
from nuham_mscheme.mscheme2 import (
    SECTORS,
    QuadBlock,
    TInt,
    _product_tensors,
    quad_blocks,
)
from nuham_mscheme.orbits import Orbit, orbit_list
from nuham_mscheme.run2 import require_clean_tree

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("runs/inputs/N3LO_EM500_srg2_hw16_emax8_e2max16.me2j.gz")
PRODUCTION_SHA256 = "f7ce10a7cdcd5760926fd5cfe25f0e69576fd0eb9930d124ec569fd2edc659e5"
DEFAULT_OUT = Path("runs/out/benstead_trace")
STORED_STATS = Path("runs/out/emax8_hw16.json")
DEFAULT_BLOCKS = ("pn:34", "pn:-34", "pp:32", "pp:-32", "nn:32", "nn:-32")
TAU_HARD = 5e-9
ORACLE_TOL = {"v": 2e-6, "tint": 1e-5, "v_plus_tint": 5e-6}
PART_LABELS = {"v": "V", "tint": "T_int,2", "v_plus_tint": "V + T_int,2"}
STORED_PARTS = {"v": "v", "tint": "tint", "v_plus_tint": "v+tint"}
SECTOR_SPECIES = {"pp": (0, 0), "pn": (0, 1), "nn": (1, 1)}
PAPER_ROWS: dict[tuple[str, int], dict[str, Any]] = {
    ("pn", 34): {"table": "VIII", "entries_per_sign": 1, "value_mev": 2.09},
    ("pn", -34): {"table": "VIII", "entries_per_sign": 1, "value_mev": 2.09},
    ("pp", 32): {"table": "VI", "entries_per_sign": 2, "value_mev": 4.18},
    ("pp", -32): {"table": "VI", "entries_per_sign": 2, "value_mev": 4.18},
    ("nn", 32): {"table": "VII", "entries_per_sign": 2, "value_mev": 4.01},
    ("nn", -32): {"table": "VII", "entries_per_sign": 2, "value_mev": 4.01},
    ("pp", 30): {"table": "VI", "entries_per_sign": 7, "value_mev": 16.64},
    ("pp", -30): {"table": "VI", "entries_per_sign": 7, "value_mev": 16.64},
}


@dataclass(frozen=True)
class BlockKey:
    """A two-body block ``(sector, 2M)``.

    Attributes
    ----------
    sector : str
        ``pp``, ``pn`` or ``nn``.
    two_m : int
        Twice the pair projection, ``M = m_p + m_q``.
    """

    sector: str
    two_m: int

    def __str__(self) -> str:
        """Return the manifest/CLI spelling, e.g. ``pn:-34``."""
        return f"{self.sector}:{self.two_m}"

    @property
    def filename(self) -> str:
        """Return the output stem, e.g. ``pn_-34``."""
        return f"{self.sector}_{self.two_m}"

    @property
    def paper(self) -> dict[str, Any] | None:
        """Return the paper row for this block, if the spec lists one."""
        return PAPER_ROWS.get((self.sector, self.two_m))


@dataclass
class Trace:
    """Everything exported for one block.

    Attributes
    ----------
    key : BlockKey
        The block.
    pair_states : list[tuple[int, int]]
        Single-particle state indices ``(p, q)``, ``p < q``, in the block order.
    parities : list[int]
        Parity of each pair state, ``+1`` or ``-1``.
    values : dict[str, numpy.ndarray]
        Full matrices for ``v``, ``tint`` and ``v_plus_tint``.
    v_records : dict[tuple[int, int], list[dict[str, Any]]]
        Per-slot V contributions, keyed by the upper-triangle slot ``(bra, ket)``.
    t_terms : dict[tuple[int, int], list[dict[str, Any]]]
        Per-slot T_int,2 direct/exchange terms, keyed by the upper-triangle slot.
    oracle : dict[str, list[float]]
        Oracle values per part, aligned with the upper-triangle slot order.
    """

    key: BlockKey
    pair_states: list[tuple[int, int]]
    parities: list[int]
    values: dict[str, np.ndarray]
    v_records: dict[tuple[int, int], list[dict[str, Any]]]
    t_terms: dict[tuple[int, int], list[dict[str, Any]]]
    oracle: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class BuildResult:
    """Result and counters of one filtered pass over the file.

    Attributes
    ----------
    traces : dict[BlockKey, Trace]
        The requested blocks.
    tuples_read : int
        Number of file quadruples read.
    tuples_used : int
        Quadruples that can reach at least one requested block.
    """

    traces: dict[BlockKey, Trace]
    tuples_read: int
    tuples_used: int


@dataclass
class OracleReport:
    """Status and diagnostics of the optional imsrg++ comparison.

    Attributes
    ----------
    status : str
        ``run``, ``not_run`` or ``not_requested``.
    reason : str or None
        Why the oracle was not run.
    image : str
        Podman image used by ``tools/run_oracle.sh``.
    work_dir : str
        Host directory holding the imsrg++ checkout and build.
    modes : dict[str, str]
        Oracle mode per exported part.
    tolerances : dict[str, float]
        Acceptance tolerance per exported part, MeV.
    max_abs_diff : dict[str, float]
        Largest absolute oracle difference per part.
    violations : list[dict[str, object]]
        Elements above tolerance.
    """

    status: str
    reason: str | None
    image: str
    work_dir: str
    modes: dict[str, str]
    tolerances: dict[str, float]
    max_abs_diff: dict[str, float] = field(default_factory=dict)
    violations: list[dict[str, object]] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Lower-case hexadecimal digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_blocks(spec: str) -> list[BlockKey]:
    """Parse a comma-separated block list such as ``pp:30,pp:-30``.

    Parameters
    ----------
    spec : str
        Comma-separated ``sector:2M`` entries.

    Returns
    -------
    list[BlockKey]
        Blocks in the given order.

    Raises
    ------
    ValueError
        If an entry is malformed or the sector is unknown.
    """
    keys: list[BlockKey] = []
    for item in spec.split(","):
        text = item.strip()
        if not text:
            continue
        sector, sep, two_m = text.partition(":")
        if not sep or sector not in SECTOR_SPECIES:
            raise ValueError(f"bad block {text!r}; expected sector:2M with sector in pp/pn/nn")
        try:
            keys.append(BlockKey(sector, int(two_m)))
        except ValueError as exc:
            raise ValueError(f"bad 2M in block {text!r}") from exc
    if not keys:
        raise ValueError("no blocks requested")
    return keys


def _pair_states(
    orbits: Sequence[Orbit], sps: Sequence[tuple[int, int, int]], key: BlockKey
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return the unique pair states and parities of one block.

    Parameters
    ----------
    orbits : Sequence[Orbit]
        File orbit list.
    sps : Sequence[tuple[int, int, int]]
        Single-particle states ``(species, orbit, 2m)`` in spec order.
    key : BlockKey
        Block to enumerate.

    Returns
    -------
    pairs : list[tuple[int, int]]
        ``(p, q)`` with ``p < q`` and total ``2m = key.two_m``.
    parities : list[int]
        Parity ``(-1)**(l_p + l_q)`` of each pair.
    """
    t1, t2 = SECTOR_SPECIES[key.sector]
    pairs: list[tuple[int, int]] = []
    for p, (sp1, _orbit1, m1) in enumerate(sps):
        if sp1 != t1:
            continue
        for q in range(p + 1, len(sps)):
            sp2, _orbit2, m2 = sps[q]
            if sp2 == t2 and m1 + m2 == key.two_m:
                pairs.append((p, q))
    parities = [-1 if (orbits[sps[p][1]].l + orbits[sps[q][1]].l) % 2 else 1 for p, q in pairs]
    return pairs, parities


def _weight(
    sector: str, bra_assign: int, ket_assign: int, values: np.ndarray
) -> tuple[np.ndarray, str]:
    """Return the kernel weight per J and a human-readable rule label.

    Parameters
    ----------
    sector : str
        ``pp``, ``pn`` or ``nn``.
    bra_assign, ket_assign : int
        Species assignment of the bra and ket pair.
    values : numpy.ndarray
        File values of shape ``(nJ, 4)``.

    Returns
    -------
    weight : numpy.ndarray
        Weight of each J in the kernel sum.
    rule : str
        Which file column or pn half-sum the weight comes from.
    """
    if sector == "pp":
        return values[:, 3], "pp column (T=1, pp)"
    if sector == "nn":
        return values[:, 1], "nn column (T=1, nn)"
    if bra_assign == ket_assign:
        return 0.5 * (values[:, 2] + values[:, 0]), "0.5*(pn + T0)"
    return 0.5 * (values[:, 2] - values[:, 0]), "0.5*(pn - T0)"


def _slot_and_signs(
    tup: Me2jTuple,
    orbits: Sequence[Orbit],
    blk: QuadBlock,
    idx: tuple[int, int, int, int],
    key: BlockKey,
    sps: Sequence[tuple[int, int, int]],
    sindex: dict[tuple[int, int, int], int],
    pidx: dict[tuple[int, int], int],
) -> tuple[int, int, int, int] | None:
    """Map one tensor element to a block slot and its two sorting signs.

    Parameters
    ----------
    tup : Me2jTuple
        File quadruple.
    orbits : Sequence[Orbit]
        File orbit list.
    blk : QuadBlock
        Kernel block holding the element.
    idx : tuple[int, int, int, int]
        Tensor indices ``(ia, ib, ic, id)``.
    key : BlockKey
        Block being exported.
    sps : Sequence[tuple[int, int, int]]
        Single-particle states.
    sindex : dict
        Map ``(species, orbit, 2m)`` to the single-particle index.
    pidx : dict
        Map a pair of single-particle indices to its block index.

    Returns
    -------
    tuple[int, int, int, int] or None
        ``(row, col, sign_bra, sign_ket)``, or ``None`` when the element only reaches the
        lower triangle of a diagonal file block.
    """
    ia, ib, ic, id_ = idx
    oa, ob, oc, od = (orbits[k] for k in (tup.a, tup.b, tup.c, tup.d))
    if key.sector == "pn":
        tb = (0, 1) if blk.bra_assign == 0 else (1, 0)
        tk = (0, 1) if blk.ket_assign == 0 else (1, 0)
    else:
        tb = tk = SECTOR_SPECIES[key.sector]

    sa = sindex[(tb[0], tup.a, 2 * ia - oa.j2)]
    sb = sindex[(tb[1], tup.b, 2 * ib - ob.j2)]
    sc = sindex[(tk[0], tup.c, 2 * ic - oc.j2)]
    sd = sindex[(tk[1], tup.d, 2 * id_ - od.j2)]

    if key.sector == "pn":
        bra_pair = (sa, sb) if tb[0] == 0 else (sb, sa)
        ket_pair = (sc, sd) if tk[0] == 0 else (sd, sc)
    else:
        bra_pair = (sa, sb) if sa < sb else (sb, sa)
        ket_pair = (sc, sd) if sc < sd else (sd, sc)
    sign_bra = 1 if sa < sb else -1
    sign_ket = 1 if sc < sd else -1
    row = pidx[bra_pair]
    col = pidx[ket_pair]

    diagonal_tuple = tup.a == tup.c and tup.b == tup.d
    if row <= col:
        return row, col, sign_bra, sign_ket
    if not diagonal_tuple:
        return col, row, sign_bra, sign_ket
    return None


def build_traces(
    path: str | Path,
    emax: int,
    e2max: int,
    keys: Sequence[BlockKey],
    tint: TInt,
) -> BuildResult:
    """Build the requested blocks from the file with full provenance.

    Only file quadruples whose two orbit pairs can carry the block's total M are
    expanded (the spec section 8 filter).

    Parameters
    ----------
    path : str or Path
        ``me2j`` file.
    emax, e2max : int
        File limits.
    keys : Sequence[BlockKey]
        Blocks to export.
    tint : TInt
        Two-body T_int parameters.

    Returns
    -------
    BuildResult
        Traces and pass counters.
    """
    orbits = orbit_list(emax)
    sps = sp_states(orbits, emax)
    sindex = {(sp, orbit, m2): i for i, (sp, orbit, m2) in enumerate(sps)}
    traces: dict[BlockKey, Trace] = {}
    pidx_by_key: dict[BlockKey, dict[tuple[int, int], int]] = {}
    for key in keys:
        pairs, parities = _pair_states(orbits, sps, key)
        pidx = {pq: i for i, pq in enumerate(pairs)}
        pidx_by_key[key] = pidx
        n = len(pairs)
        traces[key] = Trace(
            key=key,
            pair_states=pairs,
            parities=parities,
            values={
                "v": np.zeros((n, n)),
                "tint": np.zeros((n, n)),
                "v_plus_tint": np.zeros((n, n)),
            },
            v_records=defaultdict(list),
            t_terms=defaultdict(list),
        )

    tuples_read = 0
    tuples_used = 0
    coefficient = -tint.hw / tint.a
    for tup in iter_tuples(path, emax, e2max):
        tuples_read += 1
        oa, ob, oc, od = (orbits[k] for k in (tup.a, tup.b, tup.c, tup.d))
        jsum_bra = oa.j2 + ob.j2
        jsum_ket = oc.j2 + od.j2
        reach = min(jsum_bra, jsum_ket)
        active = [key for key in keys if abs(key.two_m) <= reach]
        if not active:
            continue
        tuples_used += 1

        blocks_v = list(quad_blocks(orbits, tup))
        blocks_t = list(quad_blocks(orbits, tup, tint=tint, include_v=False))
        diagonal_tuple = tup.a == tup.c and tup.b == tup.d
        t_direct, t_exchange = _product_tensors(oa, ob, oc, od, coefficient)
        js = list(range(tup.jmin, tup.jmin + len(tup.values)))
        cab_all = [cg_table(oa.j2, ob.j2, 2 * j) for j in js]
        ccd_all = [cg_table(oc.j2, od.j2, 2 * j) for j in js]

        for blk_v, blk_t in zip(blocks_v, blocks_t, strict=True):
            sector = SECTORS[blk_v.sector]
            candidates = [key for key in active if key.sector == sector]
            if not candidates:
                continue
            weights, rule = _weight(sector, blk_v.bra_assign, blk_v.ket_assign, tup.values)
            mb2 = blk_v.mb2[:, :, None, None]
            mk2 = blk_v.mk2[None, None, :, :]
            base = blk_v.valid_b[:, :, None, None] & blk_v.valid_k[None, None] & (mb2 == mk2)
            for key in candidates:
                keep = base & (mb2 == key.two_m)
                if not keep.any():
                    continue
                trace = traces[key]
                pidx = pidx_by_key[key]
                for ia, ib, ic, id_ in np.argwhere(keep):
                    ia_i, ib_i, ic_i, id_i = int(ia), int(ib), int(ic), int(id_)
                    mapped = _slot_and_signs(
                        tup, orbits, blk_v, (ia_i, ib_i, ic_i, id_i), key, sps, sindex, pidx
                    )
                    if mapped is None:
                        continue
                    row, col, sign_bra, sign_ket = mapped
                    sign = sign_bra * sign_ket
                    x_v = float(blk_v.x[ia_i, ib_i, ic_i, id_i])
                    x_t = float(blk_t.x[ia_i, ib_i, ic_i, id_i])
                    val_v = sign * x_v
                    val_t = sign * x_t
                    val_sum = val_v + val_t

                    trace.values["v"][row, col] += val_v
                    trace.values["tint"][row, col] += val_t
                    trace.values["v_plus_tint"][row, col] += val_sum
                    if row != col and not diagonal_tuple:
                        trace.values["v"][col, row] += val_v
                        trace.values["tint"][col, row] += val_t
                        trace.values["v_plus_tint"][col, row] += val_sum

                    slot = (min(row, col), max(row, col))
                    contributions: list[dict[str, Any]] = []
                    for jj, jval in enumerate(js):
                        cg_bra = float(cab_all[jj][ia_i, ib_i])
                        cg_ket = float(ccd_all[jj][ic_i, id_i])
                        contribution = sign * float(weights[jj]) * cg_bra * cg_ket
                        if contribution == 0.0:
                            continue
                        contributions.append(
                            {
                                "quadruple": [tup.a, tup.b, tup.c, tup.d],
                                "j": int(jval),
                                "file_values": [float(v) for v in tup.values[jj]],
                                "sign_bra": int(sign_bra),
                                "sign_ket": int(sign_ket),
                                "bra_assign": int(blk_v.bra_assign),
                                "ket_assign": int(blk_v.ket_assign),
                                "cg_bra": cg_bra,
                                "cg_ket": cg_ket,
                                "weight": float(weights[jj]),
                                "weight_rule": rule,
                                "contribution": contribution,
                            }
                        )
                    rec_sum = math.fsum(float(rec["contribution"]) for rec in contributions)
                    if abs(rec_sum - val_v) > 1e-13:
                        raise RuntimeError(
                            f"V provenance does not add up for {key} slot {slot}: "
                            f"{rec_sum!r} vs {val_v!r}"
                        )
                    trace.v_records[slot].extend(contributions)

                    direct = sign * float(t_direct[ia_i, ib_i, ic_i, id_i])
                    exchange = sign * float(t_exchange[ia_i, ib_i, ic_i, id_i])
                    if sector == "pp" or sector == "nn":
                        t_contribution = direct - exchange
                        t_rule = "direct - exchange"
                    elif blk_v.bra_assign == blk_v.ket_assign:
                        t_contribution = direct
                        t_rule = "direct"
                    else:
                        t_contribution = -exchange
                        t_rule = "-exchange"
                    if abs(t_contribution - val_t) > 1e-13:
                        raise RuntimeError(
                            f"T_int provenance does not add up for {key} slot {slot}: "
                            f"{t_contribution!r} vs {val_t!r}"
                        )
                    if direct != 0.0 or exchange != 0.0:
                        trace.t_terms[slot].append(
                            {
                                "quadruple": [tup.a, tup.b, tup.c, tup.d],
                                "bra_assign": int(blk_v.bra_assign),
                                "ket_assign": int(blk_v.ket_assign),
                                "sign_bra": int(sign_bra),
                                "sign_ket": int(sign_ket),
                                "coefficient": float(coefficient),
                                "direct": direct,
                                "exchange": exchange,
                                "rule": t_rule,
                                "contribution": t_contribution,
                            }
                        )

    return BuildResult(traces=traces, tuples_read=tuples_read, tuples_used=tuples_used)


def _ut_values(mat: np.ndarray) -> list[float]:
    """Return the upper-triangle values including the diagonal, row-major."""
    n = mat.shape[0]
    return [float(mat[i, j]) for i in range(n) for j in range(i, n)]


def summarize(trace: Trace) -> dict[str, Any]:
    """Compute the block summary from the exported pair states and values only.

    Parameters
    ----------
    trace : Trace
        Block trace.

    Returns
    -------
    dict
        Counts, 1-norms, maximum magnitude and the paper row, per part.
    """
    parities = trace.parities
    n = len(parities)
    structural_ut = sum(1 for i in range(n) for j in range(i, n) if parities[i] == parities[j])
    by_parity = {
        "+1": sum(1 for p in parities if p == 1),
        "-1": sum(1 for p in parities if p == -1),
    }
    parts: dict[str, Any] = {}
    for part, mat in trace.values.items():
        ut = _ut_values(mat)
        diag = [float(mat[i, i]) for i in range(n)]
        lam_ut = math.fsum(abs(v) for v in ut)
        lam_diag = math.fsum(abs(v) for v in diag)
        parts[part] = {
            "label": PART_LABELS[part],
            "nnz_ut_tau0": sum(1 for v in ut if abs(v) > 0.0),
            "nnz_ut_tau5e-9": sum(1 for v in ut if abs(v) > TAU_HARD),
            "nnz_diag_tau0": sum(1 for v in diag if abs(v) > 0.0),
            "nnz_diag_tau5e-9": sum(1 for v in diag if abs(v) > TAU_HARD),
            "lambda_ut": lam_ut,
            "lambda_full": 2.0 * lam_ut - lam_diag,
            "lambda_diag": lam_diag,
            "max_abs": max((abs(v) for v in ut), default=0.0),
        }
    return {
        "n_pair_states": n,
        "n_pair_states_by_parity": by_parity,
        "structural_ut": structural_ut,
        "paper": trace.key.paper,
        "parts": parts,
    }


def _close(value: float, expected: float, rel: float = 1e-12) -> bool:
    """Return whether two floats agree to a relative tolerance."""
    scale = max(abs(value), abs(expected))
    return abs(value - expected) <= rel * scale if scale else True


def check_stored_values(
    stored_path: Path,
    summaries: dict[BlockKey, dict[str, Any]],
    key_order: Sequence[BlockKey],
) -> dict[str, Any]:
    """Compare summaries with the stored production statistics, failing loudly.

    Parameters
    ----------
    stored_path : Path
        Stored manifest, normally ``runs/out/emax8_hw16.json``.
    summaries : dict[BlockKey, dict]
        Summaries produced by :func:`summarize`.
    key_order : Sequence[BlockKey]
        Blocks to check, in output order.

    Returns
    -------
    dict
        Report with checked blocks and any comparisons that were skipped.

    Raises
    ------
    RuntimeError
        If a stored count or 1-norm differs.
    """
    if not stored_path.exists():
        raise RuntimeError(
            f"production input requested but the stored statistics {stored_path} are missing"
        )
    data = json.loads(stored_path.read_text())
    errors: list[str] = []
    checked: list[str] = []
    for key in key_order:
        skey = f"{key.sector}:{key.two_m}"
        for part, stored_part in STORED_PARTS.items():
            stored = data.get("parts", {}).get(stored_part, {}).get("blocks", {}).get(skey)
            if stored is None:
                continue
            got = summaries[key]["parts"][part]
            checked.append(f"{skey}:{part}")
            if int(summaries[key]["structural_ut"]) != int(stored["structural_ut"]):
                errors.append(
                    f"{skey}:{part} structural_ut: "
                    f"{summaries[key]['structural_ut']} != {stored['structural_ut']}"
                )
            for count_key in (
                "nnz_ut_tau0",
                "nnz_ut_tau5e-9",
                "nnz_diag_tau0",
                "nnz_diag_tau5e-9",
            ):
                stored_key = {
                    "nnz_ut_tau0": ("nnz_ut", "0.0"),
                    "nnz_ut_tau5e-9": ("nnz_ut", "5e-09"),
                    "nnz_diag_tau0": ("nnz_diag", "0.0"),
                    "nnz_diag_tau5e-9": ("nnz_diag", "5e-09"),
                }[count_key]
                expected = int(stored[stored_key[0]][stored_key[1]])
                if int(got[count_key]) != expected:
                    errors.append(f"{skey}:{part} {count_key}: {got[count_key]} != {expected}")
            for value_key in ("lambda_ut", "lambda_full", "lambda_diag", "max_abs"):
                expected = float(stored[value_key])
                if not _close(float(got[value_key]), expected):
                    errors.append(f"{skey}:{part} {value_key}: {got[value_key]!r} != {expected!r}")
    if errors:
        raise RuntimeError(
            "exported summary does not reproduce the stored statistics:\n  " + "\n  ".join(errors)
        )
    return {"path": str(stored_path), "status": "ok", "checked": checked}


def _oracle_available() -> tuple[bool, str, Path]:
    """Return whether podman and the imsrg++ build are available."""
    work = Path(os.environ.get("WORK", "/tmp/nuham-imsrg"))
    if shutil.which("podman") is None:
        return False, "podman not found", work
    if not (work / "imsrg" / "build" / "libIMSRG.so").exists():
        return False, f"imsrg++ build missing under {work}", work
    return True, "", work


def _query_line(
    bra_pair: tuple[int, int],
    ket_pair: tuple[int, int],
    sps: Sequence[tuple[int, int, int]],
    orbits: Sequence[Orbit],
) -> str:
    """Return one driver query for a slot, with states in increasing pair order."""
    fields = []
    for sp_index in (*bra_pair, *ket_pair):
        species, orbit, m2 = sps[sp_index]
        o = orbits[orbit]
        tz = -1 if species == 0 else 1
        fields.append(f"{o.n} {o.l} {o.j2} {tz} {m2}")
    return "2  " + "  ".join(fields)


def run_oracle(
    path: str | Path,
    emax: int,
    e2max: int,
    traces: dict[BlockKey, Trace],
    key_order: Sequence[BlockKey],
    tint: TInt,
) -> OracleReport:
    """Run imsrg++ for every exported slot and attach the values to the traces.

    Parameters
    ----------
    path : str or Path
        ``me2j`` file.
    emax, e2max : int
        File limits.
    traces : dict[BlockKey, Trace]
        Blocks to query.
    key_order : Sequence[BlockKey]
        Query order.
    tint : TInt
        T_int parameters, used as ``HW`` and ``A`` for the ``trel`` modes.

    Returns
    -------
    OracleReport
        Status, tolerances and differences.
    """
    modes = {"v": "file", "tint": "trel", "v_plus_tint": "file+trel"}
    ok, reason, work = _oracle_available()
    if not ok:
        return OracleReport(
            status="not_run",
            reason=reason,
            image="imsrg-dev",
            work_dir=str(work),
            modes=modes,
            tolerances=ORACLE_TOL,
        )

    orbits = orbit_list(emax)
    sps = sp_states(orbits, emax)
    slots_by_key: dict[BlockKey, list[tuple[int, int]]] = {}
    queries: list[str] = []
    for key in key_order:
        trace = traces[key]
        n = len(trace.pair_states)
        slots = [(i, j) for i in range(n) for j in range(i, n)]
        slots_by_key[key] = slots
        for i, j in slots:
            queries.append(_query_line(trace.pair_states[i], trace.pair_states[j], sps, orbits))

    report = OracleReport(
        status="run",
        reason=None,
        image="imsrg-dev",
        work_dir=str(work),
        modes=modes,
        tolerances=ORACLE_TOL,
    )
    driver = ROOT / "tools" / "run_oracle.sh"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for part, mode in modes.items():
            tin = tmp / f"queries_{part}.in"
            tout = tmp / f"values_{part}.out"
            tin.write_text("\n".join(queries) + "\n")
            env = {**os.environ, "MODE": mode, "HW": str(tint.hw), "A": str(tint.a)}
            try:
                subprocess.run(
                    [str(driver), str(path), str(emax), str(e2max), str(tin), str(tout)],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"imsrg++ oracle failed in mode {mode}:\n{exc.stderr}\n{exc.stdout}"
                ) from exc
            values = np.atleast_1d(np.loadtxt(tout))
            if values.size != len(queries):
                raise RuntimeError(f"oracle returned {values.size} values, expected {len(queries)}")
            pos = 0
            for key in key_order:
                trace = traces[key]
                nslots = len(slots_by_key[key])
                trace.oracle[part] = [float(v) for v in values[pos : pos + nslots]]
                pos += nslots

    worst = {part: 0.0 for part in modes}
    violations: list[dict[str, object]] = []
    for key in key_order:
        trace = traces[key]
        for pos, (i, j) in enumerate(slots_by_key[key]):
            for part in modes:
                got = float(trace.values[part][i, j])
                oracle = float(trace.oracle[part][pos])
                diff = abs(got - oracle)
                worst[part] = max(worst[part], diff)
                if diff > ORACLE_TOL[part]:
                    violations.append(
                        {
                            "block": str(key),
                            "bra": i,
                            "ket": j,
                            "part": part,
                            "value": got,
                            "oracle": oracle,
                            "abs_diff": diff,
                            "tolerance": ORACLE_TOL[part],
                        }
                    )
    report.max_abs_diff = worst
    report.violations = violations
    return report


def _pair_state_payload(
    index: int,
    sp_pair: tuple[int, int],
    parity: int,
    sps: Sequence[tuple[int, int, int]],
    orbits: Sequence[Orbit],
) -> dict[str, Any]:
    """Return the exported description of one pair state."""
    particles = []
    for sp_index in sp_pair:
        species, orbit, m2 = sps[sp_index]
        o = orbits[orbit]
        particles.append(
            {
                "state_index": sp_index,
                "species": "p" if species == 0 else "n",
                "n": o.n,
                "l": o.l,
                "2j": o.j2,
                "2m": m2,
                "orbit": orbit,
                "e": o.e,
            }
        )
    return {
        "index": index,
        "p": sp_pair[0],
        "q": sp_pair[1],
        "parity": parity,
        "particles": particles,
    }


def _element_payload(
    trace: Trace,
    i: int,
    j: int,
    slot_pos: int,
) -> dict[str, Any]:
    """Return one upper-triangle element with all parts and provenance."""
    element: dict[str, Any] = {
        "bra": i,
        "ket": j,
        "bra_parity": int(trace.parities[i]),
        "ket_parity": int(trace.parities[j]),
    }
    for part, mat in trace.values.items():
        value = float(mat[i, j])
        entry: dict[str, Any] = {
            "value": value,
            "value_text": f"{value:.12e}",
            "ut_nonzero_tau0": bool(abs(value) > 0.0),
            "ut_nonzero_tau5e-9": bool(abs(value) > TAU_HARD),
        }
        if part in trace.oracle:
            oracle = float(trace.oracle[part][slot_pos])
            entry["oracle"] = oracle
            entry["oracle_abs_diff"] = abs(value - oracle)
        else:
            entry["oracle"] = None
            entry["oracle_abs_diff"] = None
        if part == "v":
            records = trace.v_records.get((i, j), [])
            total = math.fsum(float(rec["contribution"]) for rec in records)
            entry["contributions"] = records
            entry["contribution_sum"] = total
            entry["contribution_residual"] = total - value
        if part == "tint":
            terms = trace.t_terms.get((i, j), [])
            total_t = math.fsum(float(term["contribution"]) for term in terms)
            entry["terms"] = terms
            entry["term_sum"] = total_t
            entry["term_residual"] = total_t - value
        element[part] = entry
    return element


def trace_payload(
    trace: Trace,
    input_path: str,
    input_sha256: str,
    emax: int,
    e2max: int,
    tint: TInt,
    oracle_report: OracleReport,
) -> dict[str, Any]:
    """Return the JSON payload of one block.

    Parameters
    ----------
    trace : Trace
        Block trace.
    input_path : str
        Input file as recorded in the manifest.
    input_sha256 : str
        Its SHA-256.
    emax, e2max : int
        File limits.
    tint : TInt
        T_int parameters.
    oracle_report : OracleReport
        Oracle status included in the payload.

    Returns
    -------
    dict
        JSON-serialisable block export.
    """
    orbits = orbit_list(emax)
    sps = sp_states(orbits, emax)
    pair_states = [
        _pair_state_payload(k, pair, trace.parities[k], sps, orbits)
        for k, pair in enumerate(trace.pair_states)
    ]
    slots = [(i, j) for i in range(len(trace.parities)) for j in range(i, len(trace.parities))]
    elements = [_element_payload(trace, i, j, pos) for pos, (i, j) in enumerate(slots)]
    return {
        "sector": trace.key.sector,
        "two_m": trace.key.two_m,
        "block": str(trace.key),
        "convention": (
            "B (docs/spec.md 4.2): unique pair states p<q, upper triangle including the diagonal"
        ),
        "input": {"path": input_path, "sha256": input_sha256},
        "file_limits": {"emax": emax, "e2max": e2max},
        "tint": {"hw": tint.hw, "a": tint.a},
        "oracle": {
            "status": oracle_report.status,
            "reason": oracle_report.reason,
            "modes": oracle_report.modes,
            "tolerances": oracle_report.tolerances,
            "max_abs_diff": oracle_report.max_abs_diff,
        },
        "pair_states": pair_states,
        "summary": summarize(trace),
        "elements": elements,
    }


def _md_value(value: float | None) -> str:
    """Format an optional value with twelve significant printed digits."""
    return "—" if value is None else f"{value:.12e}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """Write the human-readable trace of one block.

    Parameters
    ----------
    path : Path
        Markdown file to write.
    payload : dict
        The block payload produced by :func:`trace_payload`.
    """
    summary = payload["summary"]
    lines = [
        f"# Block {payload['block']}",
        "",
        f"Input: `{payload['input']['path']}`  ",
        f"SHA-256: `{payload['input']['sha256']}`  ",
        f"File limits: emax = {payload['file_limits']['emax']}, "
        f"e2max = {payload['file_limits']['e2max']}  ",
        f"T_int,2: hbar*omega = {payload['tint']['hw']} MeV, A = {payload['tint']['a']}  ",
        f"Oracle: {payload['oracle']['status']}"
        + (f" ({payload['oracle']['reason']})" if payload["oracle"]["reason"] else ""),
        "",
        "## Summary",
        "",
        f"Pair states: {summary['n_pair_states']} "
        f"(+ parity {summary['n_pair_states_by_parity']['+1']}, "
        f"- parity {summary['n_pair_states_by_parity']['-1']}); "
        f"structural upper-triangle slots: {summary['structural_ut']}.",
        "",
        "| part | nnz UT τ=0 | nnz UT τ=5e-9 | λ_ut | λ_full | λ_diag | max |value| |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for part in ("v", "tint", "v_plus_tint"):
        st = summary["parts"][part]
        lines.append(
            f"| {st['label']} | {st['nnz_ut_tau0']} | {st['nnz_ut_tau5e-9']} | "
            f"{st['lambda_ut']:.12e} | {st['lambda_full']:.12e} | "
            f"{st['lambda_diag']:.12e} | {st['max_abs']:.12e} |"
        )
    paper = summary.get("paper")
    if paper is not None:
        lines.extend(
            [
                "",
                f"Paper row: Table {paper['table']}, {paper['entries_per_sign']} entries "
                f"per sign of M, {paper['value_mev']} MeV.",
            ]
        )

    lines.extend(
        [
            "",
            "## Pair states",
            "",
            "| index | parity | p: species n l 2j 2m orbit e | q: species n l 2j 2m orbit e |",
            "| ---: | ---: | --- | --- |",
        ]
    )
    for pair in payload["pair_states"]:
        cells = []
        for particle in pair["particles"]:
            cells.append(
                f"{particle['species']} {particle['n']} {particle['l']} {particle['2j']} "
                f"{particle['2m']} {particle['orbit']} {particle['e']}"
            )
        lines.append(f"| {pair['index']} | {pair['parity']:+d} | {cells[0]} | {cells[1]} |")

    lines.extend(["", "## Elements", ""])
    for element in payload["elements"]:
        lines.extend(
            [
                f"### slot ({element['bra']}, {element['ket']})  "
                f"parities ({element['bra_parity']:+d}, {element['ket_parity']:+d})",
                "",
                "| part | value | τ=0 | τ=5e-9 | oracle | |difference| |",
                "| --- | ---: | :---: | :---: | ---: | ---: |",
            ]
        )
        for part in ("v", "tint", "v_plus_tint"):
            entry = element[part]
            lines.append(
                f"| {PART_LABELS[part]} | {_md_value(entry['value'])} | "
                f"{'yes' if entry['ut_nonzero_tau0'] else 'no'} | "
                f"{'yes' if entry['ut_nonzero_tau5e-9'] else 'no'} | "
                f"{_md_value(entry['oracle'])} | {_md_value(entry['oracle_abs_diff'])} |"
            )
        records = element["v"]["contributions"]
        lines.extend(["", "**V record contributions**", ""])
        if records:
            lines.append(
                "| quadruple | J | T=0 | nn | pn | pp | s_bra | s_ket | assign | "
                "CG_bra | CG_ket | weight | contribution |"
            )
            lines.append(
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | "
                "---: | ---: | ---: | ---: |"
            )
            for rec in records:
                fv = rec["file_values"]
                assign = f"({rec['bra_assign']},{rec['ket_assign']})"
                quad = rec["quadruple"]
                lines.append(
                    f"| ({quad[0]},{quad[1]},{quad[2]},{quad[3]}) | {rec['j']} | "
                    f"{fv[0]:.12e} | {fv[1]:.12e} | {fv[2]:.12e} | {fv[3]:.12e} | "
                    f"{rec['sign_bra']:+d} | {rec['sign_ket']:+d} | {assign} | "
                    f"{rec['cg_bra']:.12e} | {rec['cg_ket']:.12e} | "
                    f"{rec['weight']:.12e} | {rec['contribution']:.12e} |"
                )
            lines.append(
                f"\nSum of contributions: {element['v']['contribution_sum']:.12e}; "
                f"residual: {element['v']['contribution_residual']:.3e}."
            )
        else:
            lines.append("No nonzero records.")

        terms = element["tint"]["terms"]
        lines.extend(["", "**T_int,2 direct/exchange terms**", ""])
        if terms:
            lines.append(
                "| quadruple | assign | s_bra | s_ket | direct | exchange | rule | contribution |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |")
            for term in terms:
                quad = term["quadruple"]
                assign = f"({term['bra_assign']},{term['ket_assign']})"
                lines.append(
                    f"| ({quad[0]},{quad[1]},{quad[2]},{quad[3]}) | {assign} | "
                    f"{term['sign_bra']:+d} | {term['sign_ket']:+d} | "
                    f"{term['direct']:.12e} | {term['exchange']:.12e} | {term['rule']} | "
                    f"{term['contribution']:.12e} |"
                )
            lines.append(
                f"\nSum of terms: {element['tint']['term_sum']:.12e}; "
                f"residual: {element['tint']['term_residual']:.3e}."
            )
        else:
            lines.append("No nonzero terms.")
        lines.append("")
    path.write_text("\n".join(lines))


def build_manifest(
    *,
    input_path: str,
    input_sha256: str,
    expected_sha256: str,
    emax: int,
    e2max: int,
    keys: Sequence[BlockKey],
    tint: TInt,
    oracle_report: OracleReport,
    build: BuildResult,
    summaries: dict[BlockKey, dict[str, Any]],
    stored_report: dict[str, Any],
    outputs: dict[str, str],
    wall_time: float,
    allow_other_input: bool,
    allow_dirty: bool,
    allow_holdout: bool,
) -> dict[str, Any]:
    """Return the run manifest of the spec section 7.

    Parameters
    ----------
    input_path : str
        Input file path.
    input_sha256 : str
        Its digest.
    expected_sha256 : str
        The production digest of the spec.
    emax, e2max : int
        File limits.
    keys : Sequence[BlockKey]
        Exported blocks.
    tint : TInt
        T_int parameters.
    oracle_report : OracleReport
        Oracle status.
    build : BuildResult
        Pass counters.
    summaries : dict[BlockKey, dict]
        Per-block summaries.
    stored_report : dict
        Stored-statistics comparison.
    outputs : dict[str, str]
        Output file names and SHA-256 digests.
    wall_time : float
        Measured wall time in seconds.
    allow_other_input, allow_dirty, allow_holdout : bool
        Permission flags used by the run.
    """
    return {
        "date": time.strftime("%Y-%m-%d"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "allow_dirty": allow_dirty,
        "allow_other_input": allow_other_input,
        "allow_holdout": allow_holdout,
        "input": {
            "path": input_path,
            "sha256": input_sha256,
            "spec_sha256": expected_sha256,
            "matches_spec": input_sha256 == expected_sha256,
        },
        "file_limits": {"emax": emax, "e2max": e2max, "lmax": emax},
        "tint": {"hw": tint.hw, "a": tint.a},
        "convention": "B (docs/spec.md 4.2): unique pair states, lambda over both triangles",
        "blocks": [str(key) for key in keys],
        "block_summaries": {str(key): summaries[key] for key in keys},
        "pass": {
            "tuples_read": build.tuples_read,
            "tuples_used": build.tuples_used,
            "filter": "j_a + j_b >= |2M| and j_c + j_d >= |2M| (doubled units)",
        },
        "stored_check": stored_report,
        "oracle": {
            "requested": oracle_report.status != "not_requested",
            "status": oracle_report.status,
            "reason": oracle_report.reason,
            "image": oracle_report.image,
            "work_dir": oracle_report.work_dir,
            "driver": "tools/run_oracle.sh",
            "modes": oracle_report.modes,
            "tolerances_mev": oracle_report.tolerances,
            "max_abs_diff_mev": oracle_report.max_abs_diff,
            "violations": oracle_report.violations,
        },
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "wall_time_seconds": wall_time,
        "outputs": outputs,
    }


def _git(*args: str) -> str:
    """Run a git command in the repository and return its stripped output."""
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _check_holdout(keys: Sequence[BlockKey], emax: int, allow_holdout: bool) -> None:
    """Refuse the pp/nn 2M = +-28 hold-out at the emax 8 production limits."""
    if allow_holdout or emax != 8:
        return
    held = [key for key in keys if key.sector in ("pp", "nn") and abs(key.two_m) == 28]
    if held:
        raise RuntimeError(
            "pp/nn blocks with |2M| = 28 are held out by docs/benstead-block-trace-spec.md; "
            "pass --allow-holdout to export them anyway"
        )


def _write_outputs(
    out_dir: Path,
    traces: dict[BlockKey, Trace],
    payloads: dict[BlockKey, dict[str, Any]],
    key_order: Sequence[BlockKey],
) -> dict[str, str]:
    """Write one JSON and one Markdown file per block and return their hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for key in key_order:
        json_path = out_dir / f"{key.filename}.json"
        md_path = out_dir / f"{key.filename}.md"
        json_path.write_text(json.dumps(payloads[key], indent=1) + "\n")
        write_markdown(md_path, payloads[key])
        outputs[json_path.name] = sha256_file(json_path)
        outputs[md_path.name] = sha256_file(md_path)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : Sequence[str], optional
        Arguments, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    start = time.perf_counter()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="me2j file")
    ap.add_argument("--emax", type=int, default=8, help="file emax")
    ap.add_argument("--e2max", type=int, default=16, help="file e2max")
    ap.add_argument("--hw", type=float, default=16.0, help="hbar*omega in MeV for T_int,2")
    ap.add_argument("--mass", type=int, default=16, help="mass number A for T_int,2")
    ap.add_argument("--blocks", default=",".join(DEFAULT_BLOCKS), help="comma-separated sector:2M")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--oracle", action="store_true", help="run the imsrg++ oracle")
    ap.add_argument("--allow-other-input", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-holdout", action="store_true")
    args = ap.parse_args(argv)

    require_clean_tree(args.allow_dirty)
    keys = parse_blocks(args.blocks)
    _check_holdout(keys, args.emax, args.allow_holdout)

    input_sha256 = sha256_file(args.input)
    if input_sha256 != PRODUCTION_SHA256 and not args.allow_other_input:
        raise RuntimeError(
            f"{args.input} has SHA-256 {input_sha256}, not the production digest "
            f"{PRODUCTION_SHA256}; pass --allow-other-input to override"
        )

    tint = TInt(args.hw, args.mass)
    build = build_traces(args.input, args.emax, args.e2max, keys, tint)
    summaries = {key: summarize(build.traces[key]) for key in keys}

    if input_sha256 == PRODUCTION_SHA256:
        stored_report = check_stored_values(STORED_STATS, summaries, keys)
    else:
        stored_report = {
            "path": str(STORED_STATS),
            "status": "skipped",
            "reason": "input is not the production file",
            "checked": [],
        }

    if args.oracle:
        oracle_report = run_oracle(args.input, args.emax, args.e2max, build.traces, keys, tint)
        if oracle_report.violations:
            raise RuntimeError(
                "oracle comparison exceeded tolerance:\n"
                + json.dumps(oracle_report.violations[:10], indent=1)
            )
    else:
        oracle_report = OracleReport(
            status="not_requested",
            reason=None,
            image="imsrg-dev",
            work_dir=str(Path(os.environ.get("WORK", "/tmp/nuham-imsrg"))),
            modes={"v": "file", "tint": "trel", "v_plus_tint": "file+trel"},
            tolerances=ORACLE_TOL,
        )

    payloads = {
        key: trace_payload(
            build.traces[key],
            str(args.input),
            input_sha256,
            args.emax,
            args.e2max,
            tint,
            oracle_report,
        )
        for key in keys
    }
    outputs = _write_outputs(args.out_dir, build.traces, payloads, keys)
    wall_time = time.perf_counter() - start
    manifest = build_manifest(
        input_path=str(args.input),
        input_sha256=input_sha256,
        expected_sha256=PRODUCTION_SHA256,
        emax=args.emax,
        e2max=args.e2max,
        keys=keys,
        tint=tint,
        oracle_report=oracle_report,
        build=build,
        summaries=summaries,
        stored_report=stored_report,
        outputs=outputs,
        wall_time=wall_time,
        allow_other_input=args.allow_other_input,
        allow_dirty=args.allow_dirty,
        allow_holdout=args.allow_holdout,
    )
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    for key in keys:
        st = summaries[key]["parts"]
        print(
            f"{key}: pairs {summaries[key]['n_pair_states']}, "
            f"V nnz_ut {st['v']['nnz_ut_tau0']}, lambda_ut {st['v']['lambda_ut']:.12e}, "
            f"T_int nnz_ut {st['tint']['nnz_ut_tau0']}, "
            f"sum lambda_ut {st['v_plus_tint']['lambda_ut']:.12e}, "
            f"oracle {oracle_report.status}"
        )
    print(f"wrote {args.out_dir} in {wall_time:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
