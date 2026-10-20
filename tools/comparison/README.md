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
