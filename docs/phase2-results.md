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

## Per M₃ block against Benstead et al. Tables IX–XII

Transcribed into `docs/data/benstead_tables_ix_xii.json` (key 2·M₃, one sign; in the PDF text each caption follows its
rows). Re-summing them with ±M₃ doubled gives PPN 548.5 + PNN 1546.1 + PPP and NNN 1.1 = 2095.6 MeV and
81,964 + 191,286 + 108 = 273,358 nonzeros, the published 2096 and 2.73e5. Compare with
`uv run python tools/compare_benstead3.py runs/out/3n/e8_hw30to16.json` (nnz and λ_ut at τ = 0). Ratios ours / theirs,
hw30to16 (hw16 differs only in the λ column, by about +4% to +8%):

| 2M₃ | PPN nnz | PPN λ | PNN nnz | PNN λ | PPP nnz |
|---|---|---|---|---|---|
| 1 | 2413 | 830 | 1157 | 339 | 4.1e5 |
| 3 | 2916 | 1141 | 1232 | 380 | 3.3e5 |
| 5 | 3378 | 1339 | 1305 | 428 | 7.9e5 |
| 7 | 3988 | 1801 | 1407 | 469 | 1.2e6 |
| 9 | 4114 | 1820 | 1463 | 515 | 5.3e5 |
| 11 | 4441 | 1913 | 1587 | 606 | |
| 13 | 3884 | 1553 | 1497 | 551 | |
| 15 | 4534 | 2005 | 1727 | 543 | |
| 17 | 2346 | 1198 | 1173 | 399 | |

NNN equals PPP. Their PPP and NNN λ are 0.00 to 0.13 MeV, so a λ ratio is not meaningful there (ours is 868 to 7520
MeV).

Findings:

1. **No block is close.** Every block has 340 to 4500 times the 1-norm and 1100 to 4500 times the nonzeros (PPP and
   NNN: 3e5 to 1e6 times the nonzeros). The ratio is not constant in M₃, so it is not an overall scale or a counting
   convention (full against upper triangle would be a factor 2).
2. **Their PPN and PNN blocks differ; ours do not.** Their PNN λ is 2.5 to 3.8 times their PPN λ, and their PNN nnz 2.0
   to 2.8 times their PPN nnz, in every M₃ block. Their PPP and NNN are equal. An isospin-invariant three-body operator
   gives PPN = PNN block by block (the T₋ symmetry tested in `tests/test_mscheme3.py`); ours agree to 2 nonzeros in
   2.4e8. So their three-body operator is charge symmetric (p↔n) but not isospin invariant, and the 3N file used here
   cannot reproduce that however it is cut.
3. **Not a magnitude threshold.** Their mean |W| per nonzero is 0.0075 MeV (PPN, M₃ = 1/2: 132.02 / 17,707; PNN:
   0.0088). The ladder reaches their nonzero count at τ ≈ 0.05 MeV, where the elements left have |W| > 0.05 and our PPN
   M₃ = 1/2 block still holds 720 MeV (hw16: 910), against their 132. Any cut that leaves as few elements as theirs
   leaves a larger mean. Their blocks have fewer and smaller elements than a cut of our operator can give.
4. **Same range of M₃.** Both have blocks up to |M₃| = 17/2 in PPN and PNN, and both fall steeply with M₃.

This is the same kind of disagreement as in two bodies (`docs/benstead-comparison.md`): the counts and norms of their
tables cannot be reproduced from the LNL file by a cut or a convention. The three-body question for the authors: which
three-body operator (force, regulator, basis conversion) and which sparsification gave Tables IX–XII, and why PPN and
PNN differ.

## Speed after batching the kernel

The first kernel built one small block per (assignment pair, M₃), about 100 elements each, so the time was Python
overhead (5.3 µs per slot). `tuple_tensor` (`mscheme3.py`) expands a file tuple into one padded tensor over (pair, M₃,
row, column) with batched matrix products, and `stats3.py` bins the magnitudes of a tuple with one pass. The old
`triple_blocks` stays as the reference. Checks: the tensor equals the old blocks element by element (max difference
2.8e-17, same set of blocks, signs included; `test_batched_tensor_equals_the_block_kernel`); the statistics still
equal the dense matrices; and the full e3max 8 sweep of the hw30to16 file reproduces the manifest above: identical
structural counts and identical nonzero counts at every τ > 0 in all 80 blocks, 1-norms to 2.4e-15 relative, max to
3e-17. The counts at τ = 0 differ by 18 in total (roundoff noise, which is why τ = 5e-9 is the robust threshold).

| | Before | After |
|---|---|---|
| e3max 6, one core | 93 s | 4.4 s |
| e3max 8, one core | about 1900 s (estimated) | 91 s (0.17 µs per slot) |
| e3max 8, 16 threads | 176 s | 20 s |

The 16 threads are 8 physical cores and give 4.7 times one core, limited by memory bandwidth, not by BLAS threads.
`tests/test_run3.py` dropped from 160 s to 15 s.

Rough exact cost with the new speed (slot counts extrapolated, the per-slot cost may grow with j by a factor of 2 to
3, and the larger files need a chunked streaming reader first):

| e3max | Slots | One core | This machine (4.7×) |
|---|---|---|---|
| 10 | about 6e9 | 17 min | 4 min |
| 12 | about 4.5e10 | 2 h | 25 min |
| 14 | about 2e11 | 9 h | 2 h |
| 16 | about 7e11 | 33 h | 7 h |

## Open

- e3max above 8 (Phase 3, the sampler) and the output shape for nuclear-rank-representation, both not decided.
