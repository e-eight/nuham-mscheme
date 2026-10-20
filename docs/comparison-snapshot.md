# Repaired comparison inputs, version 1

The exporter in `tools/comparison/export_snapshot.py` assembles eight pinned
manifests into the six F8/P6/P8 × physical-3N-cutoff-8/12 cases. The scientific
contract is qchem-to-nuclear ADRs 0007–0008, derivation 10 and its
`notes/experiments/2026-10-19-comparison-input-repair.md`.

This is a local metadata export. It does not expand coefficients, run on
DeltaAI, or compute a resource cost. `snapshot-spec-v1.json` pins the required
raw bytes, source layouts, use limits, kinetic settings, producer commits and
evidence. Updating those pins requires a new reviewed input version.

## Reproduce

Use a clean checkout of the exporter revision recorded in the output manifest:

```bash
python tools/comparison/export_snapshot.py export \
  --commit FULL_EXPORTER_COMMIT \
  --inputs /path/to/nuham-mscheme \
  --science /path/to/qchem-to-nuclear \
  --out /path/to/new-bundle-directory
python tools/comparison/export_snapshot.py validate /path/to/new-bundle-directory
```

Export needs the four coefficient files under the input project's `runs/inputs/`,
the six historical manifests there, and the two recovered cluster outputs
and prior evidence at the paths in the pinned spec. Required files never skip.
It hashes all four coefficient sources locally. `--out` must not exist.

Validation reconstructs `inputs.json` from the bundled raw manifests, checks
all invariants and compares the entire result to the pinned specification.
It needs only the bundle and this exporter revision; the coefficient binaries
are not bundled. Thus portable validation authenticates the stored input
statistics and interpretation, rather than rerunning the expansion or upstream
generation. The export manifest separately records source-file verification.

## Schema and numerical conventions

- `sources/`: byte-for-byte raw manifests, including every diagnostic threshold.
- `evidence/`: pinned source-restriction results, producer diffs, cluster run
  manifests, pilot verification and explanatory records. Referenced full job
  outputs remain in their original run directories.
- `spec.json`: a copy that must match the specification in the pinned checkout.
- `inputs.json`: policy, units, common settings, limitations, raw identities,
  20 component records and six case mappings.
- `manifest.json`: export revision, environment, resolved parameters, source
  hashes, elapsed time, status and output hashes. Written before work.

Every component stores `blocks` and sector/`all` totals at tau=0, including
diagonal and structural counts and norms. Missing raw diagonal totals are
derived from blocks. Integers sum exactly; floats use `math.fsum`. Stored raw
totals are checked with rtol=1e-12, atol=1e-12 MeV before export. The snapshot
reports measured residuals; last-bit differences from raw total summation
are expected within that tolerance.

`norm_scale=16/15` converts intrinsic one-body T to full T at A=16. Only
zero-threshold counts are carried through the conversion. Positive-threshold
ladders remain unconverted in the original manifests. Full T uses `v`;
intrinsic T uses the measured `v+tint` coefficient sum, never the sum of
separate norms. The separate two-body `tint` component is diagnostic.

Each case has full/intrinsic variants with component references ordered by
rank, summed `lambda_full` and `loaded_upper_count`. These are coefficient
inputs, not oracle costs or system-register widths. P8/cutoff-8 has an explicit
W8 alias to F8/cutoff-8 while retaining its own NN, kinetic and projected-space
identity. Physical E=12 remains distinct from effective P6/P8 triple filters
9/11. The historical qchem `checks/nuham_inputs.py` must retain its pinned hash.

## Verification and review

`tests/test_comparison_snapshot.py` uses direct Hermitian matrix enumeration,
an independent HO shell-degeneracy sum, all-input reconstruction and deliberate
mutations of counts, norms, sectors, thresholds, kinetic mappings, aliases,
source/use limits, missing data, evidence and hashes.

Run all 21 tests, including required real-input integration, with:

```bash
QCHEM_SNAPSHOT_ROOT=/path/to/bundle PYTHONPATH=src \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_comparison_snapshot.py
```

Without the bundle environment variable, integration tests explicitly skip.
Such a run is insufficient to accept this snapshot.

Pre-export numerical review found no changes to expansion code. Historical
NN accumulator changes only parameterize sector/bin sizes with identical NN
defaults; other NN numerical kernels are unchanged. Compared with the later
E12 manifest producers, the selected branch differs in `src` only by the
driver's early dirty-tree check. The exporter enforces a clean pin itself.
Prior independent-oracle and platform/worker evidence is retained, not
relabelled as a fresh full-case expansion test.

Historical NN generation/runtime and complete historical BLAS records remain
incomplete. Upstream float32 lab storage and text precision, scenario
truncations, and downstream physical accuracy/preparation/cost claims remain
explicit limitations. Zero-threshold support describes the stored numerical
operator; it does not establish exact physical zeros.
