# Phase 0 spec: files, m-scheme convention, λ and nnz

Status: **approved** (user, 2026-10-09). Phase 1 code exists and the two-body parts are verified (imsrg++ oracle, invariance tests). The three-body reader and expansion kernel exist (Phase 2, steps 1–4, 2026-10-10) and pass the invariance tests and the imsrg++ oracle below. Originally: nothing here was verified numerically
unless tagged **[V]**. Tags: **[S]** read from a source document, **[D]** derived here and not yet tested,
**[?]** open. Sources: [nuham-jax `docs/spec-3-lab-transform-me2j.md`](https://github.com/e-eight/nuham-jax/blob/main/docs/spec-3-lab-transform-me2j.md) (me2j), [`spec-5d-lab-transform-and-me3j.md`](https://github.com/e-eight/nuham-jax/blob/main/docs/spec-5d-lab-transform-and-me3j.md)
(me3j), [qchem-to-nuclear `CONVENTIONS.md`](https://github.com/e-eight/qchem-to-nuclear/blob/main/CONVENTIONS.md), [`checks/benstead_table2.py`](https://github.com/e-eight/qchem-to-nuclear/blob/main/checks/benstead_table2.py), and Benstead et al. 2607.21563
(arXiv PDF read 2026-10-09: Eqs. 24–29, App. B.4, Tables VI–XII).

## 1. Inputs

The user supplies `emax`, `e2max`, `e3max` and `lmax = emax`. The files do not store them. A wrong value
silently misaligns the stream, so the reader must check the value count (below) against the file.

### 1.1 `me2j` (gzip text) **[S]**
- Line 1 is a free-text header. Then Fortran `10f16.8`: 10 values per line, MeV, 8 decimals.
- Orbit list: e = 0..emax, then l (same parity as e), then j = l−½ before l+½. 15 orbits at emax 4, 45 at emax 8.
- Loop: `a ≥ b`, `c ≤ a`, `d ≤ (c==a ? b : c)`, skip if e_a+e_b or e_c+e_d > e2max, skip if
  (−1)^(l_a+l_b+l_c+l_d) = −1. Then J = max(|ja−jb|,|jc−jd|)/2 .. min(ja+jb, jc+jd)/2.
- Per (a,b,c,d,J), four values in the order **T=0 (Tz=0), nn (T=1), pn (T=1, Tz=0), pp (T=1)**.
- Normalization is the Darmstadt "un-normalised" one. pp and nn are multiplied by √2 for each of a==b and
  c==d. The pn columns carry the √2 automatically. Forbidden entries are written as 0.0.
- The file holds V only, with no kinetic term and no Hcm. 4×(tuple count) values. 27,336 values at emax 4,
  e2max 8.

### 1.2 `me3j` / `me3j.gz` **[S]**
- Same text layout, one header line. Orbits as above. Reader and writer need **e3max ≤ e2max**. Never read a
  file with e3max > e2max.
- Loop (spec-5d §6.2): a ≥ b ≥ c with e_a+e_b, e_a+e_c, e_b+e_c ≤ e2max and e_a+e_b+e_c ≤ e3max; likewise
  d ≥ e ≥ f; bra triple ≥ ket triple lexicographically; equal parity. Then j12, j45, doubled J (inside),
  then five isospin entries in the fixed order (T12,T45,2T) = (0,0,1), (0,1,1), (1,0,1), (1,1,1), (1,1,3).
- Value = fully antisymmetrised, un-normalised element ⟨abc;J12 T12|V Σ_P sgn(P)|def;J45 T45⟩, in the
  (ab)c pair-coupled basis, M_J and M_T independent. The factor 6 is hard-coded in NuHamil (spec-5d §5.2).
  Entries with a==b and (J12+T12) even, or a==b==c with T=3/2 and j<3/2, are written as 0.0.
- Value counts: 24,890 (4/8/4), 456,320 (6/12/6), 5,199,790 (8/16/8), 273,181,225 (10/20/12),
  6,367,911,165 (14/28/16). A file at e3max contains every smaller e3max as a filter (prefix in `a`, plus a
  skip on e_a+e_b+e_c).
- Precision: 8 decimals (±5e-9 MeV). NuHamil's default single precision is noisier, about 1e-6 relative.
  imsrg++ drops |V| ≤ 1e-6 on reading. Both matter for nnz (§4.3).

### 1.3 Consistency checks done on every read
Value count equals the loop count; no nonzero value at the forbidden entries; (0s½)³ block is rank one
(spec-5d §5.3). Any failure stops the run.

## 2. The m-scheme basis

- Single-particle state p = (species, n, l, 2j, 2m_j). Protons t_z = −½, neutrons +½ (NuHamil, Benstead).
- Order: protons first, then neutrons. Inside a species by e, then l, then j, then m_j ascending
  (CONVENTIONS.md). N = 330 + 330 = 660 states at emax 8 (45 orbits per species).
- Integers 2j, 2m, 2t_z everywhere. CG, 6j via exact rationals where tested, float64 in production.
- Unique pair states are p < q in this order, r < s likewise. Unique triples p < q < r.

## 3. Conversion from J-coupled elements to m-scheme **[D]**

Target quantity: the antisymmetrised, normalised element ⟨pq|V|rs⟩ = ⟨0|a_q a_p V a†_r a†_s|0⟩ for p<q, r<s
(and ⟨pqr|W|stu⟩ for triples), in MeV. Hermitian and real.

**Rule (uniform for 2B and 3B).** Order the orbits of each state by the file's orbit index, carrying the sign of
the sorting permutation (the states are antisymmetric, so this is exact). Then the m-scheme element is the
Clebsch–Gordan sum of the file value with **no extra normalization factor**:

    2B:  ⟨pq|V|rs⟩ = Σ_{J,T} ⟨ja ma jb mb|J M⟩ ⟨jc mc jd md|J M⟩ ⟨½ ta ½ tb|T Tz⟩ ⟨½ tc ½ td|T Tz⟩ V_file^{JT}(ab,cd)
    3B:  ⟨pqr|W|stu⟩ = Σ_{J12 J45 J T12 T45 T} CG_J12 CG_J CG_J45 CG_J' CG_T12 CG_T CG_T45 CG_T' W_file(...)

(pp and nn have only T=1 terms; for pp/nn the isospin CGs are 1.)

Why no normalization factor: the file value already carries the √2 per identical-orbit pair (2B) or is the
un-normalised ⟨abc|VΣsgn P|def⟩ (3B). A state a†_p a†_q|0⟩ with a==b has overlap √2·CG with the normalised
coupled state, which cancels the 1/√2 of the normalised element. For three distinct orbits
⟨A|V|A⟩ = ⟨abc|V Σ sgn P|def⟩ exactly. **[D]**: derived by hand here for pp/nn and for distinct orbits; the
a==b pn case and coinciding-orbit triples follow the same argument but need the oracle.

**Charge dependence is only partly recoverable.** The pn columns are half-sums of pn-basis elements
(spec-3 §4.4): me_00 and me_10 are ½(T(ap bn|cp dn)+T(an bp|cn dp) ∓ T(ap bn|cn dp) ∓ T(an bp|cp dn)). The file
cannot separate the two terms in each pair. The m-scheme pn element must follow imsrg++'s
`Set_pn_TBME_from_iso` ([`ReadWrite.cc:928-944`](https://github.com/ragnarstroberg/imsrg/blob/master/src/ReadWrite.cc#L928-L944)), which assumes charge independence inside each pair.
The 3N file is isospin-coupled with no charge dependence at all. So the m-scheme Hamiltonian from these
files is exactly charge-independent in the 3N part and pn-symmetric in the 2N part. **[?]** Read
`Set_pn_TBME_from_iso` and confirm in Phase 1. This is not a bug, but any report must say it.

**Three-body kernel, verified (Phase 2 steps 1–3, 2026-10-10) [V].** `mscheme3.py` implements the rule for every
species assignment of the three slots, keeps one slot assignment per unique state p<q<r (increasing single-particle
index between slots of the same orbit) and multiplies by the sorting sign. Tests (`tests/test_mscheme3.py`), run on
the real LNL file (hw30to16, cut to emax 2, e2max 4, e3max 4, which contains the coinciding-orbit triples) and on
random synthetic files:
- Each unique triple state appears exactly once, and each matrix element is written by exactly one file tuple (or its
  mirror).
- The matrix is symmetric, and J₋ and T₋ commute with it: 1.2e-8 and 1.8e-8 MeV on the real file (its 8-decimal
  rounding, values up to 0.4), 1e-9 on synthetic files. A random file does not do this for b==c triples, because
  the file values of a physical operator are antisymmetric under the exchange of equal orbits. The synthetic
  generator therefore projects random values onto the antisymmetric subspace.
- Elements agree with an explicit sympy Clebsch–Gordan sum, state by state.
- The streaming statistics equal those of the dense matrices (λ, diagonal λ, nnz, upper-triangle nnz, max, slot count
  per sector and M₃).
- The result does not depend on whether protons carry t_z = −½ or +½: flipping all t_z multiplies the coupled
  isospin states by a phase that depends only on T, which bra and ket share.
- Not tested by symmetry: the overall scale of the file values (the factor 6) and the correspondence of the file's
  (J12, T12) coupling labels with ours, because a rescaling or a relabelling of an invariant operator is invariant
  too. These are settled by the imsrg++ oracle, below.
- `structural` counts the slots inside file tuples (equal parity and a common J), not every M-conserving slot.
- Check by eye (e3max 4 and 6, hw30to16, τ = 5e-9): PPP = NNN and PPN = PNN in nnz and λ, as charge independence
  demands. At e3max 6: 2.73e7 structural upper-triangle slots, 2.717e7 nonzero, λ_ut 1.41e5 MeV, max |W| 0.457 MeV,
  97 s on one core.

**Three-body kernel against imsrg++, verified (Phase 2 step 4, 2026-10-10) [V].** `tools/imsrg_mscheme_3b.cc` reads
the same file with `ReadWrite::Read_Darmstadt_3body` and evaluates `UnitTest::GetMschemeMatrixElement_3b` (an
independent recoupling code that takes the six states in any order). `tests/test_imsrg_oracle3.py` compares random
elements, stratified over the coincidence pattern of the bra and ket orbit triples (distinct, a==b, b==c, a==b==c: all
16 combinations), over all four sectors and random M₃, on the LNL hw30to16 file cut to (4, 8, 4), (6, 12, 6) and used
whole at (8, 16, 8). Result: 2304 elements at (6, 12, 6) and 768 at (8, 16, 8), of which 1890 and 560 exceed 1e-3 MeV:
maximum difference 3.8e-9 and 7.3e-9 MeV (imsrg++ stores floats), and the ratio of the two values is 1.000000 for every
element above 1e-2 MeV. So the rule of this section holds for three bodies as written: **no extra normalisation, the
factor 6 of the file is already the antisymmetrised-state normalisation**, coinciding orbits are handled correctly,
and the isospin coupling labels and phases are read as the file intends.

Conservation (all exact, used as slot-count filters and as tests): total M (pair M₂, triple M₃), total T_z,
parity.

## 4. Quantities and conventions

### 4.1 Benstead's definitions, as read from the paper
- Eq. 1 (via [nrrep `notes/conventions-sources.md`](https://github.com/e-eight/nrrep/blob/main/notes/conventions-sources.md)): H = Σ_{p<q, r<s} V_pqrs a†p a†q a_r a_s,
  restricted sums. V_pqrs is the antisymmetrised normalised element up to a sign from the a_r a_s order,
  which does not change |V| or nnz.
- Eqs. 24–29: Prep amplitudes √(|V|/λ)·ω with ω² = 2 for (pq)<(rs), 1 for (pq)=(rs), 0 below. So
  **λ = Σ_p≤q θ²|t_pq| + Σ_(pq)≤(rs) ω²|V| + Σ_(pqr)≤(stu) γ²|W| = the sum over all ordered pairs of unique
  pair states, both triangles** (and likewise for triples). The text after Eq. 29 writes the same thing as
  Σ_pq|t| + Σ_pqrs|V| + Σ_pqrstu|W|.
- d (their nnz) = number of nonzero elements with (pq) ≤ (rs), diagonal included, one-, two- and three-body.
- Typos in the paper (Eq. 29 reuses the 2B diagonal condition; Tables IX–XII use the 2B header for 3B). They
  do not change the definitions above.

### 4.2 What this package computes (primary convention "B")

    λ₁ = Σ_{p,q} |t_pq|                           (both triangles; no 1/2 on the diagonal, see below)
    λ₂ = Σ_{p<q, r<s} |⟨pq|V|rs⟩|                (both triangles)
    λ₃ = Σ_{p<q<r, s<t<u} |⟨pqr|W|stu⟩|          (both triangles)
    nnz₂ = #{(pq) ≤ (rs) : |V| > τ}              (upper triangle incl. diagonal)
    nnz₃ = #{(pqr) ≤ (stu) : |W| > τ}
    max|V|, max|W|

Also reported, so the consumer can pick: the upper-triangle sum Σ_{(pq)≤(rs)} |V| without the ω² weight
(`lambda_ut`), and the diagonal part. Split by sector (pp, pn, nn; PPP, PPN, PNN, NNN) and by M₂ (M₃) block,
M and −M separately. The one-body λ and nnz come from the one-body operator the consumer wants (kinetic or
T_int, derivation 02 of qchem-to-nuclear), defined in Phase 1. The one-body convention "halves a diagonal
one-body done as a QROM" (Benstead) is a consumer choice, not computed here.

**Decision (2026-10-09, user): use T_int.** imsrg++ does the same: `Hbare += Trel_Op` with
`Trel = KineticEnergy_Op − TCM_Op` ([`imsrg_util.cc:743`](https://github.com/ragnarstroberg/imsrg/blob/master/src/imsrg_util/imsrg_util.cc#L743); `TCM_Op` has the one-body part p²/(2mA) = ½(2n+l+3/2)ħω/A
and a two-body part from T_cm − T_rel). That is T_int = (1−1/A)Σp²/2m − (1/Am)Σ_{i<j}p_i·p_j, with a single
nucleon mass `M_NUCLEON` (the proton/neutron mass correction is a separate optional operator,
`nucleon_mass_correction`). **[S]** Consequences: the me2j and me3j files hold V only, so this package must
generate both the one-body and the two-body T_int pieces itself (HO closed forms), and they depend on A and ħω,
which are inputs, not file contents. The two-body T_int term is an extra source of m-scheme nonzeros and λ, and
is reported separately from the V part.

### 4.3 Nonzero threshold τ
A value is "nonzero" at threshold τ. Report nnz at **τ ∈ {0, 5e-9, 1e-6}**: exact nonzero, the file's print
precision (CG sums of 8-decimal numbers produce cancellation noise near 1e-8), and imsrg++'s read cutoff.
An nnz without its τ is not a result. Tolerances follow CONVENTIONS.md: sums of more than 1e6 terms use
`math.fsum`; counts are Python `int`.

### 4.4 Filters
emax ≤ file emax, e2max, e3max ≤ file values (inclusive), and the Nmax pair-energy cut of derivation 03
(matched cutoffs). Applied on the fly, not by rewriting the file.

## 5. Check of Benstead's tables (done 2026-10-09)

Re-summing Tables VI–XII from the PDF text **[V, arithmetic on printed values]**:

| Quantity | Rows treated as | Result | Published total |
|---|---|---|---|
| λ₂B | each ±M row counted twice, M=0 once | PP 7.38e5, PN 4.435e6, NN ≈ 7.4e5 (implied) → 5.91e6 | 5.91e6 |
| nnz₂B | same | PP 1.209e6, PN 4.843e7 → 5.08e7 | 5.08e7 |
| λ₃B | same | PPN 548, PNN 1546, PPP+NNN ≈ 1 → 2095 | 2096 |
| nnz₃B | same | PPN 8.20e4, PNN 1.91e5 → 2.73e5 | 2.73e5 |

So **a "±M" row is one sign and the block exists for both signs**, and the published totals are sums over
all M. The tables' λ column is labelled Σ_pqrs|V| and the nnz column "upper-diagonal". Whether the λ column
is the full sum (our λ₂) or only the upper triangle (our `lambda_ut`) cannot be told from the printed text.
That is the first thing the Phase 1 comparison must decide, by computing both and seeing which one matches
for the same files.

Also visible: mean |W| per nonzero is 0.008 MeV, and PPP = NNN has 54 nonzeros in total (11, 11, 3, 1, 1 per
sign). These are the numbers the 2000× gap rests on ([qchem-to-nuclear `notes/problem.md`](https://github.com/e-eight/qchem-to-nuclear/blob/main/notes/problem.md)).

## 6. Mapping to the consumers

- [`checks/benstead_table2.py`](https://github.com/e-eight/qchem-to-nuclear/blob/main/checks/benstead_table2.py) takes `LAM` (a single MeV number), `D` (one count), and `N` (660). The spec
  maps: `LAM = λ₁+λ₂+λ₃` (convention B), `D = nnz₁+nnz₂+nnz₃` (convention B, one τ), `N = 660`. Its placeholders
  (1e4 MeV one-body λ, one-body nnz ≤ N(N+1)/2 = 218,130) are replaced by Phase 1 output. The script is not
  changed by this package.
- qchem-to-nuclear rows 1, 3, 5 and K3: nnz per sector and per M₂/M₃ block, matched-Nmax filter.
- nuclear-rank-representation: unique-pair-state two-body tensor ⟨pq|V|rs⟩ per pair-M block (convention B is
  exactly their "Restricted" sum). **[?]** I have not read their needs beyond `conventions-sources.md`. Their
  tensor shape and ordering should be confirmed with the user before Phase 1 fixes the output format.

## 7. Output and provenance

All statistics (λ, nnz at every τ, max|V|, per sector and per block) are computed in the streaming pass, as
each block is built, and written to the manifest. Stored blocks are a secondary output and are never the
source of a statistic. Blocks are written with no magnitude cutoff, as the upper triangle (pq) ≤ (rs) plus the
pair list, so any τ can be recounted later. The 3N analogue is per-M₃ blocks over unique triples, and its
statistics come from the exact sweep or the sampler.


One JSON manifest per run, plus arrays only when asked: git commit (clean tree required for production runs),
input file path and SHA-256, emax/e2max/e3max/lmax, filters, τ values, sampler seed and stream (Phase 3),
package versions. Results keyed by (body rank, sector, M block).

## 8. Open questions for review

1. Is the tables' λ column the full sum or the upper-triangle sum (§5)? Decided by computation in Phase 1.
2. Does `Set_pn_TBME_from_iso` match the pn rule of §3? Read it, then test against imsrg++ `GetMschemeMatrixElement_2b`.
3. Coinciding-orbit 3N triples and the factor 6: the coinciding-orbit rule passes the J₋/T₋ and antisymmetry tests (§3), and the factor 6 is settled by the imsrg++ 3b oracle (ratio 1.000000, §3).
4. ~~One-body operator~~ Decided: T_int (§4.2). Still open: A and ħω as run inputs, and whether to report the T_int two-body part inside or beside the V part.
5. Which hbar-omega and file Benstead used is unknown (qchem-to-nuclear assumes 16 MeV). Not needed for the code.
6. Output layout for nuclear-rank-representation (§6).
