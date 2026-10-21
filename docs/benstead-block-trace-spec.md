# Spec: two-body block trace for the Benstead comparison

Status: **approved** (user, 2026-10-20); implementation by deepcode, check 6 by an independent coder.
Requested by: qchem-to-nuclear `notes/experiments/2026-10-20-benstead-discrepancy-plan.md`, Stage 2 (reduced).
Budget: **two hours** of preparation in total, including verification (the plan's pre-email cap). If the cap is
reached, record what is missing and stop.

## 1. Purpose

Export, for a few tiny two-body m-scheme blocks, every matrix element with its full provenance, in a form that:

1. someone else can rebuild the block's count and 1-norms from, without the streaming accumulator (`stats2.py`);
2. can be sent to Benstead et al. next to their own list for the same block.

This is a diagnostic export. It does not change the production kernel (`mscheme2.py`), the statistics
(`stats2.py`), the driver (`run2.py`) or any stored result. It tests no hypothesis about the authors' conversion.

## 2. Input

| Item | Value |
| --- | --- |
| File | `runs/inputs/N3LO_EM500_srg2_hw16_emax8_e2max16.me2j.gz` |
| SHA-256 | `f7ce10a7cdcd5760926fd5cfe25f0e69576fd0eb9930d124ec569fd2edc659e5` (as in `runs/out/emax8_hw16.json`) |
| File limits | emax = 8, e2max = 16 |
| Producer | nuham-jax (`runs/inputs/gen.log`), not original NuHamil. Record the nuham-jax commit if it can be recovered; otherwise say it is unknown |
| Operator | N3LO EM500 NN with Coulomb, SRG 2 fm⁻¹, ħω = 16 MeV, V only (spec §1.1) |
| T_int,2 | `TInt(hw=16.0, a=16)`: −(ħω/A) Σ_{i<j} π_i·π_j (spec §4.2, derivation 02 of qchem-to-nuclear) |

The trace must refuse any other file hash unless given an explicit override flag, which is then written to the
manifest.

## 3. Blocks

A block is (sector, 2M) with sector ∈ {pp, nn, pn} and M = m_p + m_q the pair projection. Both parities are in
one block, as in the paper and in `stats2.py`.

| Block | Status | Paper (per sign of M) | Ours, stored (`runs/out/emax8_hw16.json`, commit `1acd02c`) |
| --- | --- | --- | --- |
| pn, 2M = +34 and −34 | required | Table VIII: 1 entry, 2.09 MeV | V: 1 entry, λ_ut 2.09203676; T_int,2: 0 entries |
| pp, 2M = +32 and −32 | required | Table VI: 2 entries, 4.18 MeV | V: 4 entries, λ_ut 1.79953, λ_full 2.17503; T_int,2: 1 entry, 4.00000; V + T_int,2: 4 entries, λ_ut 5.79953 |
| nn, 2M = +32 and −32 | required | Table VII: 2 entries, 4.01 MeV | V: 4 entries, λ_ut 1.91614, λ_full 2.30838; T_int,2: 1 entry, 4.00000; V + T_int,2: 4 entries, λ_ut 5.83177 |
| pp, 2M = +30 and −30 | optional, only within the cap | Table VI: 7 entries, 16.64 MeV | V: 23 of 25 slots, λ_ut 8.58455 |
| pp and nn, 2M = ±28 | **do not export** | held out for later validation (plan, Stage 4) | — |

The stored values above are the gate in §6.1. Export both signs of M: the second sign is a free invariant check
(§6.3), and the paper's rows are read per sign.

## 4. What to export

### 4.1 Pair states

For each block, the list of unique pair states p < q (spec §2 order: protons first, then by orbit index, then m
ascending), each with:

- index in the block (0-based, in spec §2 order of (p, q));
- for each particle: species (p/n), n, l, 2j, 2m, file orbit index, and e = 2n + l;
- parity (−1)^(l_p + l_q).

### 4.2 Elements

Every slot (bra, ket) of the block with bra index ≤ ket index (the upper triangle, diagonal included), whether or
not the element is zero. For each slot and each part in {V, T_int,2, V + T_int,2}:

- the signed value ⟨pq|O|rs⟩ in MeV, float64, printed with `%.12e`;
- `ut_nonzero_tau0`: |value| > 0; `ut_nonzero_tau5e-9`: |value| > 5e-9 MeV;
- the imsrg++ oracle value (§5) and the absolute difference.

V + T_int,2 is summed element by element before taking absolute values (plan, error policy).

### 4.3 Provenance of each V element

Each element is a sum over file records. For each contributing record:

- the file quadruple (a, b, c, d) as stored (a ≥ b, file loop of spec §1.1), J, and the four file values in column
  order (T = 0, nn, pn T = 1, pp);
- the orbit-sorting signs of bra and ket (the `pair_index` sign in `dense2.py`), and which pn species assignment
  (`bra_assign`, `ket_assign` of `QuadBlock`) the slot came from;
- the two J Clebsch–Gordan coefficients ⟨j_a m_a j_b m_b|J M⟩ and ⟨j_c m_c j_d m_d|J M⟩;
- the weight the kernel applies: the pp or nn column; for pn, ½(v_pn + v_T0) for equal species assignments and
  ½(v_pn − v_T0) for exchanged ones (`mscheme2.py`);
- the signed contribution of the record. The contributions must add up to the exported V value (§6.2).

For T_int,2: the direct and exchange terms of `_product_tensors` for the slot, with the coefficient −ħω/A, so that
value = direct − exchange in pp/nn, and the pn terms as in `quad_blocks`.

### 4.4 Block summary

Per block and part: number of pair states (by parity), structural upper-triangle slots, upper-triangle count at
τ = 0 and τ = 5e-9, λ_ut, λ_full, λ_diag and max |value|, all computed **from the exported element list only**,
plus the paper's row for comparison. λ_full = 2 λ_ut − λ_diag.

## 5. Oracle

imsrg++ through `tools/run_oracle.sh`, the same driver and tolerances as `tests/test_imsrg_oracle.py`:

- MODE=file for V: |difference| < 2e-6 MeV;
- MODE=trel with HW=16, A=16 for T_int,2: |difference| < 1e-5 MeV;
- MODE=file+trel for V + T_int,2: |difference| < 5e-6 MeV.

Queries are the slots of §4.2 in the driver's input format (`n l j2 tz2 m2`, tz2 = −1 for protons). The driver
returns ⟨pq|O|rs⟩ in imsrg++'s a_p a_q order; map it to our sign convention the same way the existing test does. If
the oracle cannot run (image or build missing), record that and continue; the export is still useful, but the
contact email must not then claim an oracle check for these blocks.

## 6. Acceptance checks

1. **Reproduces the stored statistics.** The §4.4 summary equals the stored block statistics in §3: counts
   exactly; λ_ut, λ_full, λ_diag to 1e-12 relative. A mismatch means the trace or the stored result is wrong; stop
   and report it before anything else.
2. **Contributions add up.** For every V element, the sum of the §4.3 contributions equals the exported value to
   1e-13 MeV.
3. **Invariants on the unmodified kernel.** Hermiticity: the kernel's ⟨rs|O|pq⟩ equals ⟨pq|O|rs⟩ for every
   exported off-diagonal slot (compute it, do not assume it). Time reversal: the +M and −M blocks have the same
   pair-state count and the same multiset of |values|. Parity: no nonzero element between pair states of different
   parity.
4. **Oracle.** Every element within the §5 tolerances, or the oracle is recorded as not run.
5. **Positive control.** pn 2M = 34: V = 2.0920 MeV to the printed digits, T_int,2 = 0.
6. **Independent rebuild.** A second implementation, written from this spec, `docs/spec.md` §§1–3 and the file
   alone (not importing `mscheme2`, `dense2` or `stats2`), recomputes the V elements of the pp and nn 2M = 32
   blocks from the file records and Clebsch–Gordan coefficients and agrees to 1e-12 MeV. It may use
   `nuham_mscheme.cg` only if it checks a few coefficients against exact rationals (e.g. `sympy`).

Checks 1–5 run in a pytest test on a small fixture and once on the production input. The fixture is the
`e4_file` of `tests/conftest.py` (original NuHamil, N3LO EM500, SRG 2.0, ħω 16, emax 4, e2max 8), using its
stretched blocks pp 2M = 16 and pn 2M = 18. Check 1 there compares with `TwoBodyStats` run on the same file.

## 7. Outputs

| What | Where |
| --- | --- |
| Element lists | `runs/out/benstead_trace/<sector>_<2M>.json` (§4.1–4.4) |
| Human-readable trace | `runs/out/benstead_trace/<sector>_<2M>.md`: pair-state table, element table with V, T_int,2, sum, oracle, and the record contributions under each element |
| Manifest | `runs/out/benstead_trace/manifest.json`: git commit and dirty flag (fail on a dirty tree unless `--allow-dirty`, as `run2.py`), input path and hash, blocks, TInt, oracle mode and image, Python and NumPy versions, wall time, output hashes |
| Code | `tools/benstead_trace.py` (diagnostic tool, like `tools/benstead_hypotheses.py`); test `tests/test_benstead_trace.py`. No changes to existing modules in `src/`; a new helper there only if covered by the test |

The JSON is the artifact that may be sent to the authors. It carries no interpretation.

## 8. Cost

- Reading: only quadruples that can reach the block matter (j_a + j_b ≥ |2M| and j_c + j_d ≥ |2M| in doubled
  units); a full pass over the 2.5e6-value file is acceptable. The related `tools/benstead_hypotheses.py` pass took
  25 s on 16 cores; a single-process filtered pass is expected to take under a minute (estimate, not measured).
- Oracle: one imsrg++ build and file read in the podman image, a few minutes (estimate).
- Measure the first run and record the wall time in the manifest before rerunning anything.

## 9. Not in scope

Three-body blocks; the M₂ = ±28 hold-out; any modified conversion or hypothesis variant; other A, ħω or
potentials; regenerating the input with original NuHamil.
