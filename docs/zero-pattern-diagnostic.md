# ARM zero-pattern diagnosis

This diagnostic preserves all source kernels and original tests at c69f815.
It investigates job 3419663's synthetic dense/streaming nnz_full disagreement
(ppn, 2M=1, 70842 versus 70844) without changing thresholds or asserting a cause.
See the companion qchem-to-nuclear diagnosis plan dated 2026-10-20.

`tools/diagnose_zeros3.py` generates seed-20261010 coefficients at (2,4,4),
retains pre-cleanup values, and proves its fixture copy matches the original
test fixture bit for bit. A lossless laptop fixture is versioned with its
metadata and SHA-256. Arrays have canonical little-endian storage; compare raw
array hashes, not NPZ container hashes, to establish coefficient identity.

For native and imported coefficients the diagnostic assembles all signed
batched entries in the dense reference's canonical state basis, rejects double
writes, and compares direct full/upper/diagonal counts against the accumulator
at every threshold. It records all zero-mask differences, Hermitian partners,
source tuples, decimal/hex values, norm residuals and full toy matrices. It
retains intermediate contractions and shape/stride metadata for up to eight
implicated tuples; remaining tuple IDs stay explicitly listed. No discrepancy
is silently treated as success: `diagnostic_complete` means evidence collection
finished, not that the compatibility gate passed.
Each trace records whether instrumented replay matches the original result
bitwise. A false replay flag is retained evidence and prevents attributing the
original discrepancy to that trace without further investigation.

Review before commit checked conventions, mapping and numerical behavior:
canonical keys and finite float64 coefficients are validated; direct integer
counts are checked at every threshold; full/diagonal norms use compensated
reference sums; duplicate writes fail; input and producer hashes are rechecked;
no threshold, producer arithmetic, or original test is changed. Tests exercise
lossless round trips, malformed fixtures, an injected count error, and an injected
Hermitian zero pair that must be localized and traced. Ruff and ty cover the
new diagnostic files. These checks validate evidence collection, not ARM physics.

The native ARM fixture will be replayed locally to complete the two-fixture,
two-environment comparison. Exact-coupling sums for reduced entries and
controlled arithmetic experiments are local follow-up; no symbolic sweep runs
in this short cluster job. The existing explicit reference rounds its terms
and cannot by itself establish exact zero.

Run locally with one thread per BLAS/OpenMP library:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/diagnose_zeros3.py --fixture tools/fixtures/zero3-laptop.npz --out runs/local-diagnostic
```

The driver rejects dirty checkouts and changed historical kernels/tests.
`--allow-dirty` is only for local development; it records the tracked diff and
script hash. Cluster runs must use the committed clean version. Manifests are
written before generation/evaluation and finalized with output hashes and
peak RSS. Runs never overwrite existing directories. There is no checkpoint
support; incomplete runs remain incomplete. No automatic retry is authorized.

Deployment: push this diagnostic branch, fetch it on DeltaAI, and create an
isolated detached checkout at its full commit SHA. From that directory submit
`sbatch tools/diagnose_zeros3.sbatch FULL_COMMIT_SHA`. The job pins HEAD and
fixture hashes. The existing virtual environment is reused, with actual
versions recorded. No package installation or change to the active producer.

The user authorized one job under bhhk-dtai-gh. DeltaAI rejected a CPU-only
test request, so the minimum GPU request is one; no GPU kernels execute.
The script requests one CPU, one GPU, 512 MiB and one minute. Native local
diagnosis measured 3.83 s and 156072 KiB RSS. The first complete native-plus-
frozen rehearsal and final validation are recorded in the science project
run artifacts. These are estimates for sizing, not a claim of ARM timing.
The earlier full test suite took roughly 1.13 times the local duration on ARM.

Check scheduler exit/state, manifest status, hashes, all mismatch reports and
trace-selection limits. A mismatch is expected diagnostic evidence. P6 and W8
compatibility remain outside this job. If additional cluster work is needed,
report the evidence and obtain authorization rather than resubmitting.
