# Numerical conventions

This records existing conventions from `docs/spec.md`, `CLAUDE.md`, and the
accepted qchem-to-nuclear comparison specification; it changes no kernel.

- Energies and matrix elements are MeV. Working coefficients are real float64;
  counts are integers. Coupling quantum numbers use doubled integers.
- Single-particle indices are zero based: protons then neutrons, orbit then m.
  Triple states have p < q < r and the associated fermionic sorting signs.
- Convention B: lambda_full includes both triangles; lambda_ut includes the
  diagonal once. Full = twice upper minus diagonal for norms and counts.
- All cutoff limits are inclusive. Source layout and retained filters are distinct.
- tau=0 counts computed nonzeros of the represented numerical operator, not
  exact physical zeros. Positive thresholds remain diagnostics. Never silently
  replace tau=0 by a positive tolerance to make a check pass.
- Existing test-specific tolerances remain unchanged. Comparison checks use
  rtol=1e-12, atol=1e-12 MeV; integer identities are exact. The independent
  imsrg++ checks have their documented float-storage tolerances.
- Oracles: small dense/reference expansion, explicit coupling sums, rotation/
  isospin invariants, reader/filter comparisons and optional imsrg++. An exact
  sum with frozen rational coefficients is distinct from a float reference.
- Seeded synthetic data use NumPy default_rng(20261010); explicit sampled
  oracle queries use seed 7. Save actual coefficient bytes for platform comparisons.
- Fixed chunks determine reduction order; worker changes must not silently
  change statistics. Record threads, NumPy/BLAS and architecture.
- Store run artifacts in unique `runs/` directories with a manifest written
  before work, then finalized with status and hashes. Production uses clean
  pinned commits. Exploratory dirty runs must preserve their diff.
- Local and DeltaAI use their existing virtual environments; record actual
  package versions. DeltaAI account is bhhk-dtai-gh. The user authorized one
  bounded zero-pattern diagnostic submission on 2026-10-20; additional jobs
  need separate authorization. No numerical diagnostics on login nodes.
- Scientific specifications live in docs/spec.md; diagnosis records in docs/.
  New conventions must be recorded when adopted, not inferred from a changed test.
