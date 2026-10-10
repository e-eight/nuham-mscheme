# nuham-mscheme: plan

Turns J-coupled nuclear matrix-element files (`me2j`, `me3j`) into the m-scheme quantities that
fault-tolerant cost models need: 1-norms (λ), nonzero counts, and maximum magnitudes.

Started 2026-10-09 as a spin-off of [qchem-to-nuclear](https://github.com/e-eight/qchem-to-nuclear) (Batch 2). Status: Phase 0 spec approved
(`docs/spec.md`, 2026-10-09); Phase 1 (two-body) done, see `docs/benstead-comparison.md` for the open
comparison with Benstead's tables; Phase 2 in progress (reader, kernel and tests done; imsrg++ oracle and sweep driver pending).

## Why a separate package

- [nuham-jax](https://github.com/e-eight/nuham-jax) is meant to be a near drop-in replacement for NuHamil. NuHamil has no
  m-scheme expansion, so this does not belong there.
- [qchem-to-nuclear](https://github.com/e-eight/qchem-to-nuclear) and [nuclear-rank-representation](https://github.com/e-eight/nrrep) are in their science phase: no
  implementation code, only brute-force spot checks in `checks/`. This package is the
  implementation; the science repos consume its numbers and cite its commit.
- Both science projects need it: qchem-to-nuclear for the chiral no-core Hamiltonian (rows 1, 3, 5
  and kill criterion K3), nuclear-rank-representation for m-scheme two-body tensors (for example
  Benstead App. B Table 3, USDB per pair-M block).

## Scope

In:
- Read `me2j` and `me3j.gz` as written by nuham-jax and NuHamil (formats: nuham-jax
  [`docs/physics-spec.md`](https://github.com/e-eight/nuham-jax/blob/main/docs/physics-spec.md), [`docs/spec-5d-lab-transform-and-me3j.md`](https://github.com/e-eight/nuham-jax/blob/main/docs/spec-5d-lab-transform-and-me3j.md); imsrg++ needs `file3e1max`,
  `file3e3max`, the same values are needed here).
- Expand to the m-scheme (proton/neutron orbitals, explicit m) and compute:
  - λ (1-norm) of the one-, two- and three-body parts, in the index convention of the consuming cost
    model (qchem-to-nuclear [`checks/benstead_table2.py`](https://github.com/e-eight/qchem-to-nuclear/blob/main/checks/benstead_table2.py): Benstead's sparse LCU, (pq) ≤ (rs) style
    upper triangle; to be fixed in Phase 0);
  - nonzero counts, split by sector (pp/pn/nn; PPP/PPN/PNN/NNN) and by pair-M block;
  - max |V|.
- Filters applied on the fly: emax, e2max, e3max (a file at e3max = 16 contains every smaller
  e3max), and the Nmax pair-energy cut of qchem-to-nuclear derivation 03 (matched cutoffs).
- 3N: exact sweep where it fits (e3max ≤ 8 at emax 8: 5.4e8 structural slots); stratified sampling
  with error bars above (up to about 1e12 slots at e3max 16).

Out for now (add only when a consumer needs it): NO2B normal ordering, MP2 amplitudes, any
Hamiltonian generation (that is nuham-jax), any quantum-circuit cost model (that stays in the
science repos as spot checks).

## Oracles

- imsrg++ `UnitTest::GetMschemeMatrixElement_1b/_2b/_3b` ([`src/UnitTest.cc`](https://github.com/ragnarstroberg/imsrg/blob/master/src/UnitTest.cc) in ragnarstroberg/imsrg): one m-scheme
  element by recoupling; independent of NuHamil and nuham-jax. Needs a small C++ driver or a
  `pyIMSRG` build (the `imsrg-dev` image has no Python bindings). Random-tuple comparison.
  Also a candidate engine for the 3N sampler, where per-element speed does not matter.
- Brute force at small emax: explicit Clebsch–Gordan sums, element by element.
- Structural slot counts from qchem-to-nuclear [`checks/count3n.py`](https://github.com/e-eight/qchem-to-nuclear/blob/main/checks/count3n.py), [`count3n_scaling.py`](https://github.com/e-eight/qchem-to-nuclear/blob/main/checks/count3n_scaling.py):
  nonzeros ≤ structural, per block.
- Exact identities: rotational invariance (M-independence of block sums where it must hold),
  antisymmetry, hermiticity, symmetry zeros (2–3% of 3N slots, [`checks/count3n_scalar.py`](https://github.com/e-eight/qchem-to-nuclear/blob/main/checks/count3n_scalar.py)).
- Targets, not oracles: Benstead App. B (λ₂B = 5.91e6 MeV, nnz₂B = 5.08e7, λ₃B = 2096 MeV,
  nnz₃B = 2.73e5); disagreement is a finding, not a bug, until shown otherwise.

## Phases

| Phase | Work | Gate |
|---|---|---|
| 0 | Spec: file formats and normalization (Darmstadt un-normalised me2j, factor 6 in me3j), m-scheme antisymmetrized convention, the λ and nnz conventions of the consumers | Spec reviewed against `benstead_table2.py` and the sister project's needs |
| 1 | 2B expansion, emax 8 (e2max 16) and emax 9; kinetic and T_int one-/two-body terms as in qchem-to-nuclear derivation 02 | imsrg++ getter on random tuples; brute force at emax ≤ 4; ħω scan (16, 20, 24) vs Benstead's λ₂B |
| 2 | 3N exact sweep, emax 8, e3max 8; sector and pair-M splits | imsrg++ getter; structural bounds; Benstead's 2.73e5 as target |
| 3 | 3N stratified sampler, e3max 10–16 | exact sweep reproduced at e3max 8 within the sampler's error bar |

Compute: unmeasured. Time Phase 1 at emax 4–6 and Phase 2 at e3max 4–6 first, and give an estimate
before any run at production size. Inputs at production size live on DeltaAI (for example the LNL
emax 8, e3max 16 file of job 3330462, 16.9 GB); local machine has 16 cores and 19 GB.

## Conventions

Follow nuham-jax: `uv`, `ruff`, `ty`, `pytest`, docstrings and type annotations everywhere, NuHamil's
physical constants. Every result written to disk carries a provenance manifest (git commit, input
file hash, filters, sampler seed).
