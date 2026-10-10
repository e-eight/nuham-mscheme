# Comparison with Benstead et al. Tables VI–VIII (two-body, per M₂ block)

Status: recorded 2026-10-09. Phase 1 target check; a disagreement is a finding, not a bug, until
shown otherwise (`PLAN.md`).

## Inputs

- Ours: `runs/out/emax8_hw16.json`, commit `1acd02c`. N3LO EM500, SRG λ = 2 fm⁻¹, ħω = 16 MeV,
  emax 8, e2max 16, V only (no T_int). Generated with nuham-jax (Coulomb on by default).
- Theirs: arXiv 2607.21563v1 App. B, Tables VI (PP), VII (NN), VIII (PN), transcribed from the PDF
  into `docs/data/benstead_tables_vi_viii.json` as `{sector: {M2: [lambda, nnz]}}`. The rows are
  single-sign (±M2), so totals double every M2 ≠ 0 row.
- Their columns: "Σ_pqrs |V|" and "number of non-zero upper-diagonal elements"
  (|V| > 0 and (pq) ≤ (rs)). No ħω, threshold or conversion detail is stated.
- Reproduce: `uv run python tools/compare_benstead.py runs/out/emax8_hw16.json`.

## Totals

Re-summing the tables (±M2 doubled) gives nnz 5.08e7 and λ 5.91e6, the published totals. Ours at
the same settings, upper triangle including the diagonal: nnz 2.65e8 (99.9 % of the structural
2.655e8), λ 8.48e6 (8.48e6 / 5.91e6 = 1.4).

## Per M₂ (ours / Benstead)

| M₂ | PP nnz | PP λ | PN nnz | PN λ |
|---|---|---|---|---|
| 0 | 41.2 | 2.13 | 3.6 | 1.41 |
| 4 | 35.0 | 1.51 | 3.7 | 1.31 |
| 8 | 17.0 | 0.93 | 3.5 | 1.13 |
| 12 | 5.1 | 0.50 | 2.8 | 0.88 |
| 14 | 3.0 | 0.38 | 2.2 | 0.77 |
| 16 | 2.0 | 0.43 | 1.6 | 0.86 |
| 17 | – | – | 1.0 | 1.00 |

NN behaves like PP. Full tables: run the script above.

## Findings

1. **Not a magnitude threshold.** A cut can only lower λ, but Benstead's λ exceeds ours for PP at
   M₂ ≥ 8 (up to 2.6×) and for PN at M₂ ≥ 11, while his nnz is smaller. Their blocks are not a
   subset of ours. The ladder scan (`lambda_ut_above` in the manifests) also finds no single τ that
   fits PP, PN and the totals together: PN at M₂ = 0 fits τ ≈ 0.035 MeV, PP needs τ ≈ 0.15 for nnz
   and ≈ 0.06 for λ. The PN fit is a coincidence.
2. **PP and NN have identical nnz in every M₂ block**, and λ(PP) and λ(NN) differ by about 0.1 %.
   Ours also have identical nnz at τ = 0 (both sectors have the same structural zeros), so this
   is not evidence of a structural rule, and an earlier version of this note wrongly said it was.
   Our λ(PP)/λ(NN) differs by 0.3–1 % with or without Coulomb (see the no-Coulomb check below).
3. **PN M₂ = 17 agrees exactly** (one element, 2.09). It equals half the file's T = 0, J = 17
   value of (8s₁₇/₂)² (−4.184), as the pn state is (|T=0⟩ + |T=1⟩)/√2. This confirms ħω = 16, the
   interaction, and our pn rule in that block.
4. **PP/NN at M₂ = 16:** ours has three pair states and four elements (diagonals 0.016, 1.231,
   0.178 and an off-diagonal 0.375, total 1.80). Theirs has two elements totalling 4.18 (PP) and
   4.01 (NN). No subset of our elements reaches 4.18 (the largest possible weighted sum is 2.18),
   so their values differ, not only their selection.

## What this does and does not show

Our side is checked independently (imsrg++ `GetMschemeMatrixElement_2b` on 57,120 elements in all
three sectors, Hermiticity, J₋ invariance). The disagreement is in their JT → M-scheme conversion
or in a different input. It is not explained by a cutoff.

## Open items

- The cause of the PP/NN (and PN M₂ ≤ 16) differences. Next step is the authors.

## Hypothesis tests on the highest PP blocks (2026-10-09)

Targets, from Table VI: M₂ = 16: 2 nonzeros, λ 4.18. M₂ = 15: 7, 16.64. M₂ = 14: 71, 101.86.
M₂ = 13: 232, 272.54. Ours (upper triangle): 4 / 1.80, 23 / 8.58, 214 / 39.1, 902 / 125.3. Run on
the emax 8, ħω = 16 file, using only tuples with both pair energies ≥ 12 (enough for M₂ ≥ 13).
Exploratory scripts, not in the repository.

| Hypothesis | Result |
|---|---|
| Column mixture: the pp column replaced by any combination of the file's four columns (T=0, nn, pn, pp) with weights in {0, ½, 1} (80 combinations) | Best fit (nn + pp) gives λ 3.67 and 17.8 at M₂ = 16, 15, near their 4.18 and 16.64, but nnz 4 and 23 against 2 and 7. No combination matches the counts. |
| Same-orbit pair states (a = b) dropped | nnz 2 at M₂ = 16, but 20, 156, 812 at M₂ = 15, 14, 13 against 7, 71, 232; λ 1.25 against 4.18. Rejected. |
| Per-element rescaling, such as √2 factors for a = b or c = d | Cannot change nnz, which is already too large at M₂ = 15, 14 and 13. Not run separately. |

Conclusion: none of the tested conversion variants reproduces their nnz pattern. The PN M₂ = 17
element agrees exactly, so the interaction and ħω are not the issue. The cause of the PP/NN
difference remains unexplained. The open items are the Coulomb question above and asking the
authors (a question set already exists in qchem-to-nuclear `notes/draft-benstead-questions.md`).

## No-Coulomb check (2026-10-09)

Their pp and nn matrices look nearly charge-symmetric, so I regenerated the emax 8, ħω = 16 file
with `--no-coulomb` (`runs/out/emax8_hw16_nocoul.json`) and repeated the comparison.

- Totals barely move: nnz_ut 2.653e8 (unchanged), λ_ut 8.505e6 (8.483e6 with Coulomb).
- PP and NN: nnz equal to the Coulomb file and still 41× (M₂ = 0) to 2× (M₂ = 16) above theirs.
  λ(M₂ = 0) is 168,602 (PP) and 169,167 (NN) against their 78,180 and 78,126. The per-M₂ λ ratios
  are the same as before to within 1.3 %.
- λ(PP)/λ(NN): ours 0.990–0.997 at M₂ = 0, 4, 8, 12, 16; theirs 1.0007, 0.9996, 1.0000, 1.0011 and
  1.0424. Ours stays about 0.3–1 % below 1 (it is charge-dependence of the N3LO force and the
  file, not Coulomb); theirs scatters around 1.
- PN is unchanged (the Coulomb term acts on pp only).

Coulomb is therefore not the cause. The tests above leave the difference unexplained.
