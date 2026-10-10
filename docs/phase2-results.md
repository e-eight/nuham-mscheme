# Phase 2 results: exact three-body sweep, emax 8, e3max 8

Date 2026-10-10. Code at commit `f24e896` (clean tree). The kernel is verified against imsrg++ (`docs/spec.md` §3:
2304 and 768 random elements, maximum difference below 1e-8 MeV, ratio 1.000000).

## Inputs and runs

Both files are LNL forces (EM500 NN with the N3LO local 3NF, c1, c3, c4, cD, cE = −0.81, −3.2, 5.4, 0.7, −0.06,
Λ_local 650, Λ_nonlocal 500, SRG 2.0), written by nuham-jax with e3max 16 and cut to emax 8, e2max 16, e3max 8 with
`python -m nuham_mscheme.me3j` on DeltaAI (login node, no batch job). Each cut holds 5,199,790 values.

| File | Input SHA-256 (cut file) | Manifest |
|---|---|---|
| `N3LO_N3LOEMlnl_hw30to16_e8_e2_16_e3_8.me3j.gz` (3N operator converted from ħω 30 to 16; primary) | `c347cbcf…89e552` | `runs/out/3n/e8_hw30to16.json` |
| `N3LO_N3LOEMlnl_hw16_e8_e2_16_e3_8.me3j.gz` (generated at ħω 16; cross-check) | `0b49bc79…3650ec` | `runs/out/3n/e8_hw16.json` |

`uv run python -m nuham_mscheme.run3 FILE --emax 8 --e2max 16 --e3max 8 --workers 16 --out MANIFEST`. Each run took
about 3 minutes on 16 workers (176 s and 178 s), 0.3 GB per process. `runs/` is not in the repository; the manifests
hold the per-(sector, 2M₃) numbers at every τ of the ladder, the SHA-256 of the input and the commit.

Convention "B" (`docs/spec.md` §4.2): unique triple states p<q<r, λ over both triangles, nnz over the upper triangle
including the diagonal. **Structural** counts the slots inside file tuples (equal parity and a common J), not every
slot allowed by M₃. The two-body and one-body parts are not included. Nothing here is a statement about T_int.

## Totals, hw30to16

| Sector | Structural slots (upper triangle) | nnz at τ = 5e-9 | λ (both triangles) | λ_ut | max \|W\| |
|---|---|---|---|---|---|
| PPP | 25,091,970 | 24,931,724 | 8.17e4 MeV | 4.10e4 | 0.313 MeV |
| PPN | 238,349,558 | 237,859,025 | 1.189e6 | 5.95e5 | 0.457 |
| PNN | 238,349,558 | 237,859,027 | 1.189e6 | 5.95e5 | 0.457 |
| NNN | 25,091,970 | 24,931,724 | 8.17e4 | 4.10e4 | 0.313 |
| all | 526,883,056 | 525,581,500 | 2.541e6 | 1.272e6 | 0.457 |

PPP equals NNN, and PPN equals PNN up to 2 of 2.4e8 counts at τ = 5e-9 (rounding noise at the threshold), as charge
independence requires. 99.8% of the structural slots are nonzero.

## Threshold ladder, all sectors

| τ (MeV) | nnz_ut hw30to16 | nnz_ut hw16 | λ_ut above τ, hw30to16 | λ_ut above τ, hw16 |
|---|---|---|---|---|
| 0 | 526,367,556 | 526,367,146 | 1.272e6 | 1.371e6 |
| 5e-9 | 525,581,500 | 525,581,617 | 1.272e6 | 1.371e6 |
| 1e-6 | 525,341,453 | 525,360,030 | 1.272e6 | 1.371e6 |
| 1e-3 | 329,908,480 | 339,389,342 | 1.181e6 | 1.284e6 |
| 1e-2 | 14,233,676 | 17,258,200 | 2.19e5 | 2.78e5 |
| 0.1 | 5,316 | 22,848 | 679 | 3,216 |

## hw30to16 against hw16

The slot pattern and the zero pattern agree: identical structural counts in every sector, and nonzero counts at
τ ≤ 5e-9 that differ by a few hundred of 5.3e8. The large elements differ. hw16 has a heavier tail: λ_ut is 8% higher
(1.371e6 against 1.272e6 MeV), max \|W\| in PPN and PNN is 0.711 against 0.457 MeV (PPP and NNN: 0.313 for both), and
there are four times as many elements above 0.1 MeV. Charge independence holds in both.

## Against Benstead et al.

Their three-body totals are λ₃ = 2096 MeV and nnz₃ = 2.73e5 (App. B, Tables IX–XII; their PPP and NNN blocks hold 54
nonzeros each). Here, at τ = 5e-9:

| | λ₃ | nnz₃ |
|---|---|---|
| Benstead et al. | 2096 MeV | 2.73e5 |
| this sweep, hw30to16 (λ_ut) | 1.27e6 MeV (about 600×) | 5.26e8 (about 2000×) |
| this sweep, hw16 (λ_ut) | 1.37e6 MeV (about 650×) | 5.26e8 (about 2000×) |

No magnitude cut closes the gap: even at τ = 0.1 MeV the hw30to16 file leaves 5,316 nonzeros and λ_ut = 679 MeV, and the
hw16 file 22,848 and 3,216 MeV. The 3N operator of Benstead et al. is far sparser than this dense LNL file. Why is not
known; this package cannot say, and the comparison per M₃ block against Tables IX–XII is not done. Like the two-body
comparison (`docs/benstead-comparison.md`), the question for the authors is which three-body operator and which
sparsification were used.

## Open

- Per-M₃ comparison with Benstead's Tables IX–XII (the manifests hold the blocks).
- e3max above 8 (Phase 3, the sampler) and the output shape for nuclear-rank-representation, both not decided.
