# P6 with three-body cutoff 8: validated pilot

Completed 2026-10-20 for the qchem-to-nuclear comparison-input repair.
This supplies the missing 3N contribution for the P6/E=8 case. It does not
complete the separate six-case input snapshot or migrate the cost model.

## Identity and deployment

- Producer/job wrapper: `09fd72f5add548f26877922ed9029cce15589d8e`, clean.
  Numerical src and the validated tests/checker are unchanged from be6b584.
- DeltaAI job **3422497**, bhhk-dtai-gh, gh003; completed **0:0**.
- Request: 3 CPUs, 1536 MiB, 1 minute, one site-required GPU reservation.
  No GPU kernels. One submission; no retry or duplicate production run.
- Source: `N3LO_N3LOEMlnl_hw30to16_e8_e2_16_e3_8.me3j.gz`.
  SHA256 `c347cbcfe5c2095b99bed6659243a112fbbbaf8af4a7c645b6888e3a3e89e552`.
- File layout **(8,16,8)**; explicit retained limits **(7,8,8)**.
  Three workers, 256 fixed chunks, single-thread BLAS, --verify-count.
- Prerequisite: W8 compatibility job 3422230, manifest SHA256
  `483b8d4c65b474ec77406d599201042bb4394516e4edaa569c8e4ced4508e01c`.
  Manifest plus all recorded gate outputs were authenticated before work.
- P6 raw output SHA256
  `5e7828bba6037864a4c8a36718c2de1e0a8817fc6a4392388a85362c25e255fe`.

## Result

These digits identify the represented numerical result; they are not a claim
of physical accuracy. Norms are in MeV, convention B (both triangles).

| Quantity | P6, 3N cutoff 8 |
| --- | ---: |
| nnz_ut at tau=0 | 522559780 |
| nnz_full at tau=0 | 1044867256 |
| nnz_diag at tau=0 | 252304 |
| structural_ut | 523022768 |
| lambda_full | 2518577.570050658 |
| lambda_ut | 1260938.1145533072 |
| lambda_diag (sum over blocks) | 3298.6590559561946 |
| max_abs | 0.45713787089379126 |

Relative to W8, retained upper nonzeros are 99.2766%, and retained lambda_full
is 99.1265%. The filter removes 3807782 upper nonzeros and
22192.625237775184 MeV of 3N lambda_full. These are 3N statistics only, not a
cost ratio or an equal-energy-accuracy comparison.

## Validation and measured cost

Four new local tests passed: full/partial/minimal support partitions, deliberate
count corruption, wrong endpoint logic, and missing/wrong gate evidence.
The physical synthetic fixture is required for Hermitian upper-count identities;
reader pseudo-data are not a physical operator. Ruff and ty checks for the
changed runner/tests passed. Clean local wrapper smoke passed in 1.39 s;
derivation-10 checks passed in 2.68 s (72 projected cases on 225 determinants).
Existing 50-test/dense/exact-CG and local imsrg++ evidence remains applicable
because no numerical producer changed.

On DeltaAI, one/three-worker smoke results agree exactly and match the previous
validated smoke. All P6 count/norm/threshold/structural invariants passed.
The independent endpoint selector examined 85415 coupled tuples and expanded
2207 discarded tuples. P6 plus that complement reconstructs every W8 count
exactly, including all positive thresholds and every block/sector total.
Largest norm reconstruction residual: 3.4924596548080444e-10 MeV absolute;
largest relative residual: 9.52561430845488e-16. Both are within the unchanged
rtol=atol=1e-12 policy (MeV absolute). This validates support accounting, not
independent coupling algebra or ideal-force accuracy.

Slurm elapsed **34 s**, peak **817216 KiB** (about 798 MiB). Wrapper elapsed
28.70 s; P6 sweep 19.45 s; complementary calculation 2.06 s; smokes 1.30/1.07 s.
Python, architecture, BLAS build and NumPy matched the authenticated gate.
All seven recorded output hashes were verified after retrieval; invariants and
reconstruction were repeated locally. No job is still running.

Remote artifacts:
`/projects/bhhk/soham/qchem-comparison-checks/p6-pilot-09fd72f/runs/p6-pilot-3422497/`.
Local evidence in qchem-to-nuclear:
`runs/2026-10-20-p6-pilot-deployment/verification.json` and
`recovered/runs/p6-pilot-3422497/manifest.json` (raw p6.json beside it).
Experiment record: `notes/experiments/2026-10-20-p6-pilot.md` in that repo.
Upstream precision and physical truncation uncertainties remain as documented
there. Next: independently validate/import the zero-threshold six-case snapshot.
