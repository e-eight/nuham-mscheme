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
| 12 | about 4.5e10 (measured 5.6e10) | 2 h | 25 min (measured) |
| 14 | about 2e11 | 9 h | 2 h |
| 16 | about 7e11 | 33 h | 7 h |

## Streaming reader for big files

`iter_tuples3` no longer loads the file. A `ValueReader` (`me3j.py`) returns the values of the requested range
(value index `v0..v1`), buffers sequential requests, and seeks otherwise: forward in a gzip stream means
decompressing without parsing (ISA-L, about 800 MB/s), and a backward seek reopens the file. A chunk is a range of bra
rows (the file is written row by row: bra triple with all kets); `chunk_rows` splits the file into 256 chunks of equal
estimated cost, and each worker takes a run of consecutive chunks, skips to its first row once and reads on. The result
is a list of per-chunk statistics merged in chunk order, so it does not depend on the worker count. The value count is
checked at the end of the stream (`check_end`), a wrong limit is caught by the forced-zero entries, and
`--verify-count` first reads the whole stream once (decompression only) to compare with the layout. Caches that grew
with the file are bounded: the spatial coefficients now depend on the j class only (a few hundred tables), the
per-triple state tables take about 20 bytes per state, and the tuple index tables are an LRU cache.

At e3max 8 the sweep (hw30to16) takes 12 s on 16 threads including `--verify-count`, 0.27 GB per process, and
reproduces the previous manifest: identical structural and nonzero counts at every τ including τ = 0, 1-norms to
2.2e-16. Layouts of the larger files (`row_plan`, `chunk_rows`): (10, 20, 12) has 3,344 triples and 2.73e8 values,
(14, 28, 16) has 13,120 triples and 6.37e9 values; with 256 chunks no chunk reads more than 1.3% of the values, and
`chunk_rows` takes 2 s. These larger files have not been read yet: the gzip skip, the parse speed (about 160 MB/s of
text per core) and the per-tuple Python overhead at 4.3e7 tuples are still to be measured.

## Exact sweeps at e3max 9 to 12 (hw30to16)

The source is the LNL hw30to16 file with emax 10, e2max 16, e3max 16 (27.0 GB, on DeltaAI). Cutting it needs only a
prefix of the stream (`filter_me3j`, run on the DeltaAI login node, not as a batch job). Cuts made there, copied here
and checked by SHA-256 (each has a `.json` provenance manifest, `count_ok` and the forced-zero check true):

| Cut (emax, e2max, e3max) | Size | Values | Tuples | Cut time |
|---|---|---|---|---|
| (10, 16, 10) | 168 MB | not recorded | not recorded | not recorded |
| (10, 16, 12) | 1.04 GB | 2.73e8 | 2.64e6 | 42 min (reads 5.49e9 source values) |

The e3max 9 and 11 sweeps use filters on the e3max 10 and 12 cuts. e3max 8 from the (10, 16, 10) cut reproduces the
direct e3max 8 file (`e8_from_e10cut.json`). All sweeps below: 16 workers, `--verify-count`, clean tree, commit
`c69f815` (e3max 9 to 12), exact, no sampling.

| e3max | Slots (upper triangle) | Nonzero at τ = 5e-9 | λ_ut (MeV) | max \|W\| (MeV) | Wall time | Memory (total / largest process) |
|---|---|---|---|---|---|---|
| 8 | 5.27e8 | 5.26e8 | 1.272e6 | 0.4571 | 12 s | |
| 9 | 1.93e9 | 1.93e9 | 3.310e6 | 0.4571 | not recorded | |
| 10 | 6.45e9 | 6.44e9 | 8.002e6 | 0.4571 | not recorded | |
| 11 | 1.98e10 | 1.98e10 | 1.827e7 | 0.4571 | 7.5 min | 4.3 GB / 0.72 GB |
| 12 | 5.61e10 | 5.61e10 | 3.951e7 | 0.4571 | 25 min | 6.7 GB / 1.6 GB |

- The slot counts follow the earlier extrapolation (1.93e9 and 6.45e9 as predicted; e3max 12 came out 25% above
  4.5e10). Growth per unit e3max: 3.7, 3.3, 3.07, 2.84. The nonzero fraction at τ = 5e-9 is 99.9% at every e3max: almost
  every structural slot is nonzero.
- max |W| does not change above e3max 8. Every element added by a larger e3max is smaller than 0.4571.
- λ_ut grows more slowly than the slot count (about 2.2 per unit e3max at the top), because the added elements are small.
- The per-slot cost at e3max 12 is 0.027 µs per slot on 16 threads (25 min for 5.6e10), against 0.023 µs at e3max 8 (12 s for 5.3e8), so the cost per
  slot has not grown with j so far.
- Revised exact-cost estimate from the e3max 12 point (slots times 2.8 to 3 per step): e3max 14 about 2.5e11 slots,
  2 to 2.5 h here; e3max 16 about 9e11, about 8 h. The sources for 14 and 16 (27 to 31 GB) do not fit on this disk
  (about 32 GB free), so those cuts and sweeps would run on DeltaAI or from a streamed copy.
- Lesson: the driver checked the dirty tree only when writing the manifest and lost a finished 10 min e3max 11 sweep once.

## Matched Nmax spaces for 16O (e-fermi 1)

`--nmax N --e-fermi 1` sets the filters (`matched_cutoffs`, docs/spec.md 5): N=6 gives (7, 8, 9) and N=8 gives (9, 10, 11).
Exact sweeps, 16 workers, clean tree, commit `2ac03a7`, upper triangle, convention B:

| Input | N | Filters | Slots | Nonzero at τ = 5e-9 | λ_ut (MeV) | max \|W\| (MeV) | Wall time |
|---|---|---|---|---|---|---|---|
| 3N, LNL hw30to16, e3max 12 cut | 6 | (7, 8, 9) | 1.47e9 | 1.47e9 | 2.662e6 | 0.4571 | 38 s |
| 3N, same cut | 8 | (9, 10, 11) | 1.68e10 | 1.67e10 | 1.612e7 | 0.4571 | 7 min |
| 2B, N3LO_EM500_srg2 hw16 emax 9 e2max 18 | 8 | (9, 10) | 2.62e7 | 2.62e7 | 1.891e6 | 10.71 | 1.5 min |

Manifests: `runs/out/3n/nmax6_O16_hw30to16.json`, `runs/out/3n/nmax8_O16_hw30to16.json`,
`runs/out/emax9_hw16_nmax8_O16.json`. The 3N filters need e3max 9 and 11, so the e3max 8 cut cannot give them; both
come from the e3max 12 cut. Their slot counts are below the full e3max 9 and 11 sweeps (1.93e9, 1.98e10) because emax
and e2max are cut as well. The 2B run has no T_int (no `--hw`/`--mass`). Only the hw16 emax 9 file was run; the hw20
and hw24 files were not.

## Open

- e3max above 8 (Phase 3, the sampler) and the output shape for nuclear-rank-representation, both not decided.
