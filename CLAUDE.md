# CLAUDE.md

`nuham-mscheme`: m-scheme λ, nonzero counts and max |V| from J-coupled `me2j`/`me3j` files.
Spun off from [qchem-to-nuclear](https://github.com/e-eight/qchem-to-nuclear) on 2026-10-09. `PLAN.md` holds the scope, oracles and phases.

- Status: Phase 0 spec approved 2026-10-09 (`docs/spec.md`). Phase 1 (two-body) done 2026-10-09 and 2026-10-10 (code in `src/nuham_mscheme/`, Benstead comparison in `docs/benstead-comparison.md`); Phase 2 (three-body) done 2026-10-10 (reader, kernel, imsrg++ oracle, sweep driver `run3.py`, exact sweeps at e3max 8 to 12 (hw30to16) and the per-M₃ Benstead comparison in `docs/phase2-results.md`); Phase 3 (e3max above 12: exact sweep or sampler) planned, not started. Do not start a phase unless the user asks.
- Not part of [nuham-jax](https://github.com/e-eight/nuham-jax), which stays a near drop-in for NuHamil.
- Conventions as in [nuham-jax `CLAUDE.md`](https://github.com/e-eight/nuham-jax/blob/main/CLAUDE.md): `uv`, `ruff`, `ty`, `pytest`, docstrings and type
  annotations, NuHamil's constants.
- Give a measured compute estimate before any run beyond a small test case.
