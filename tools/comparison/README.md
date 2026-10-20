# Repaired W8 compatibility gate

This Git-tracked runner continues the accepted qchem-to-nuclear step-3 plan.
All producer kernels remain byte-identical to c69f815. The tests separate
independent element accuracy from exact counting of the batched coefficients.
Neither the strict tau=0 policy nor a numerical tolerance changes.

Run from a clean checkout, passing its full commit explicitly:

```bash
mkdir -p runs
sbatch tools/comparison/compatibility.sbatch FULL_COMMIT
```

The runner refuses dirty or mismatched revisions and source/fixture hashes.
It writes an outer manifest before work; the producer writes its usual output
manifest. Every stage must pass before the next starts:

1. Fifty reader, 3N, driver, diagnostic and exact-oracle tests, zero skips.
2. Real W8 source restricted to (2,4,4), one and three workers, 256 chunks.
   Require exact worker equality and compatibility with the local reference.
3. Full W8 (8,16,8), three workers, 256 chunks. Require all integer counts
   exactly equal to the reference; norm/max rtol=atol=1e-12 (MeV absolute).
   Zero-threshold upper-triangle count must be 526367562.
4. Recheck input and producer identity, finalize status and output hashes.

Source layout is (8,16,8) throughout, with --verify-count. No generation or
package installation. P6 is a separate follow-up after this gate is reviewed.
Full W8 is a regression, not an independent oracle of physical accuracy.
Independent small-element and symmetry checks remain in the tests; local
imsrg++ validation is documented in the science repo, not claimed here.

Use bhhk-dtai-gh, ghx4, one node/task, three CPUs, one site-required GPU,
2 GiB and four minutes. Prior ARM suite took 134 s and Slurm peak was about
823 MiB; diagnostic suite adds seconds, and local W8 took 18 s with four
workers. Three workers are sufficient for the worker smoke and full baseline;
2 GiB covers simultaneous processes and four minutes covers the measured
suite plus new regressions and W8. Record measured usage after this run.
The GPU is reserved only because DeltaAI rejects CPU-only submissions.
The user authorized minimal necessary jobs and Git deployment on 2026-10-20.

The reference JSON files are unchanged local records at 6cf23b0, with their
original provenance adjacent. Their SHA256SUMS are checked before computation.
The wrapper also checks that src has not changed from diagnosed c69f815.
The diagnostic CLI itself remains historical: its original-test guard means
it should be run at its recorded diagnostic commit, not this repaired revision.

Retrieve manifest, JSON outputs, XML, logs and Slurm accounting; check every
recorded hash and both empty difference reports. A running manifest after
scheduler termination is incomplete. There are no checkpoints: any approved
retry uses a fresh output directory, never overwrites evidence. Do not run
numerical checks on a login node; monitor scheduler status sparingly.


## P6 pilot after compatibility job 3422230

`p6_pilot.py` and `p6_pilot.sbatch` run the one missing point with explicit
use limits (7,8,8) and source layout (8,16,8), three workers, 256 fixed chunks.
Pass the full deployed commit to sbatch. The wrapper authenticates the previous
remote gate manifest and outputs, enforces unchanged numerical src/validated
tests/checker, records actual environment, and repeats the short real-source
one/three-worker smoke before the pilot. Its --smoke-only mode is for local
orchestration validation; it does not compute P6 or accept a pilot result.

The extra discarded-support calculation independently enumerates all one-,
two-, three-slot energy bounds and excludes a tuple when either endpoint fails.
Kept plus removed must reconstruct every W8 count exactly, additive norms to
rtol=atol=1e-12 MeV, and maxima by maximum. The small test covers full retention,
partial retention and the minimal space, plus count corruption and incorrect
endpoint logic. Use a physical Hermitian synthetic fixture: the reader's
pseudo-data do not support additive upper counts derived by halving full/diag.
The initial test caught that fixture mismatch; no tolerance was loosened.

Resources: bhhk-dtai-gh, ghx4/projects, 3 CPUs, 1536 MiB, 1 minute and one
site-required GPU reservation. W8 at this worker count took 19.91 s, and the
whole earlier job peaked at 1105600 KiB. This subset plus the small complement,
smoke and provenance is budgeted at roughly 30 s. Record actual pilot usage.
There is no checkpoint or automatic resubmission. Pin and deploy through Git,
write logs under ignored runs/, and guard the submission receipt against reuse.

Review: no numerical producer/test changes. Source/use limits and tau=0 are
explicit; the complement has disjoint coefficient support and is a bookkeeping
check, not a second coupling oracle. All four species and both triangles are
retained. Exact integers remain exact; float tolerances are unchanged. Previous
physical/source precision caveats continue to apply. Result acceptance requires
scheduler success, passed outer manifest, output hashes and local revalidation.
