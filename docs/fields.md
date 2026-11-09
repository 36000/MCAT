# Bundle fields: flux, committors and endpoint filtering

MCAT's maps are products of two solutions that it normally multiplies and discards.  Keeping
them (`segment_bundles(..., keep_fields=True)`, `mcat.fields`) gives every bundle-conditioned
expectation as a sum over the solution columns: where the bundle's streamlines go through
each voxel, how many cross any cut, where the walks from a voxel end, and how a bundle
changes when its endpoint rule changes.  This document explains the quantities, what they
are good for, and what was tried with them on a test-retest cohort.

## 1. The two fields

A streamline from a seed is a pair of half-walks on the chain's (voxel, direction class)
states.  For a bundle with include ROIs, inclusion-exclusion turns "passes every include
ROI" into a signed sum over *avoid sets* X (subsets of the include ROIs that the walk must
not touch); with the endpoint kinds k (start side, end side, both valid), a bundle is C
columns, one per (X, k), each with a sign.  Per column:

- **completion** `beta_c(s)`: the probability that a walk continuing from state `s` avoids
  X, stays out of the exclude ROIs, and is absorbed in an endpoint of kind k.  One backward
  solve, `beta = M (r + Q beta)` with `r = R absw_c` the probability of a valid absorption
  on the next step and `M` the taboo mask.  It is the same for every seed.
- **occupancy** `S_c(s)`: the expected number of arrivals at `s` of the half-walks launched
  from every seed, each launch weighted by its partner half's completion probability (so
  that the pair, not the half, satisfies the bundle).  One forward solve from the launch
  mass `pi0_c`, `S = pi0 + M Q^T S`.

With `sign_c` the inclusion-exclusion sign:

| quantity | per state `s` | meaning |
|---|---|---|
| visits | `n(s) = sum_c sign_c S_c(s) beta_c(s)` | the density, per direction class; `density_t` adds the streamline's step 0 at its seed |
| edge flux | `F(s, s') = sum_c sign_c S_c(s) Q(s, s') beta_c(s')` | expected bundle-streamline steps along the transition `s -> s'` |
| absorption | `A(s) = sum_c sign_c S_c(s) r_c(s)` | expected bundle-streamline steps from `s` into a valid endpoint |
| launch | `L(s) = sum_c sign_c pi0_c(s) beta_c(s)` | bundle-streamline mass entering `s` on the launch step |

These are the reactive quantities of transition path theory for the event "the walk
belongs to the bundle", on an absorbing chain with inclusion-exclusion.  `BundleFields`
holds `S`, `beta`, `launch` (`pi0`) and `a0` (mass absorbed on the launch step) with the
plan's taboo sets, signs and endpoint weights, and computes the rest.

## 2. Flux

**Conservation.**  At every state, `L + inflow = n = outflow + A`; `BundleFields.conservation()`
reports the largest per-voxel residual relative to the expected streamlines (at the solve
tolerance when the series converged).  The flux is therefore a steady current: inject one
unit of tracer per seed launch, absorb it on leaving white matter, and `n` is the tracer
concentration, `F` the current.

**Cuts.**  Summed over the edges crossing any surface that every bundle walk crosses once,
`F` is the bundle's expected number of streamlines; a wiggle across the surface adds its
re-crossings.  So the bundle's cross-section is the surface of minimum area carrying the
whole flux, and the density integrated over a slab of unit thickness is the streamline count.

**Edge flux per voxel.**  `edge_flux()` reduces `F` to voxel -> neighbour edges (26 columns in
`geometry.OFFSETS` order); `through_flux()` is the flux leaving the voxel (equal to its
visits).  `flux_orientation()` gives the principal direction of the flux tensor
`sum_n F_n d_n d_n^T` and its coherence: the direction *this bundle's* streamlines take
through the voxel, independent of the other bundles using the same fODF lobes.  The
visitation ODF (`geometry.visitation_odf`) is the same information at full angular
resolution; the tensor is its second moment.

`mcat.run(..., flux=True)` (CLI `--flux`) writes `_desc-throughflux_map.nii.gz` and
`_desc-edgeflux_map.nii.gz` (26 volumes) per bundle and records the conservation residuals
in the summary; the fields themselves are not kept in `run` (three `n_states x C` float64
arrays per bundle), use `segment_bundles(keep_fields=True)` directly for them.

## 3. Committors

`committors(targets)` replaces each column's endpoint kind by its intersection with a
subset of the valid endpoints (one extra backward solve per column and subset) and returns,
per voxel, the visits by bundle streamlines whose walk from there ends in that subset.
Divided by the visits it is the committor: the fraction of the bundle's traffic through the
voxel that finishes in that endpoint region.  Subsets that partition the valid endpoints
give committors that sum to one.  `endpoint_sides([start_roi, end_roi])` is the common case.

This answers "where do the walks through here end" exactly, from the existing solve.  It
does **not** give the bundle under a changed endpoint rule: the occupancy was computed with
each launch weighted by its partner half's completion under the *old* rule, so restricting
the endpoints changes the forward side as well.  Re-segment (`restrict_endpoints` +
`segment_bundles`) for that; the committor mass is an upper bound on the kept mass.  The
same holds for any new taboo: on the arcuate, a local-contrast flux pruning re-solved on
the stored fields kept 72% of the paths, the exact re-run 47%.

## 4. Endpoint filtering

The one streamline-free cleaning rule that behaved well: restrict a bundle to the walks
ending in its dominant endpoint regions.

**Patches.**  `endpoint_patches(endpoints)` segments an endpoint map (expected bundle
endpoints per absorbing voxel) into patches: a marker watershed of the blurred map, basins
merged when the saddle between them is at least 80% of the lower peak (trx-shape's tufts
rule), basins under 2% of the mass merged into their best-connected neighbour.  Each side of
the bundle (start / end ROI, or `core.split_ends`) is patched separately.  Patches of one
subject's own map move between sessions; patches of the maps of many subjects pooled in a
template space, then mapped back, were used for the evaluation below and are more stable.
Which patch is the "right" one is anatomy: the largest share per session is a poor rule
when two patches are of similar size (section 5).

**Filter.**  `restrict_endpoints(spec, start=patch_mask, end=patch_mask)` and re-segment.
The result is exact: a bundle whose streamlines both pass the ROIs and end in the chosen
patches.  The ratio of expected streamlines, the *kept fraction*, is the share of the
bundle's mass that was ending elsewhere.

## 5. Evidence: ds008825 test-retest

Twenty-two subjects scanned twice (the HBCD protocol, 2 mm isotropic grid), three pyAFQ
bundles (left arcuate ARC_L, right cingulum CGC_R, left corticospinal CST_L), endpoint
patches pooled over the 45 sessions in MNI space, the largest-share patch kept on each side.
Raw bundles against endpoint-filtered ones (the chain of the prototype, before this port;
scripts in the mat-tufts repository).

| | ARC_L raw / filtered | CGC_R raw / filtered | CST_L raw / filtered |
|---|---|---|---|
| expected streamlines, mean | 295 / 228 | 102 / 82 | 94 / 25 |
| kept fraction, mean (range) | 0.77 (0.69-0.85) | 0.80 (0.69-0.85) | 0.27 (0.20-0.35) |
| ICC(1) of the count | 0.996 / 0.996 | 0.962 / 0.964 | 0.990 / 0.929 |
| within-subject difference, median | 1.7% / 1.9% | 6.5% / 7.3% | 4.3% / 6.3% |
| core-mask Dice between sessions, median | 0.963 / 0.965 | 0.957 / 0.953 | 0.969 / 0.968 |
| log-density correlation, median | 0.992 / 0.992 | 0.987 / 0.987 | 0.989 / 0.989 |
| kept fraction, session 1 vs 2 | r = 0.99 | r = 0.97 | r = 0.65 |

- For the arcuate and the cingulum the filter removes about a fifth of the mass and leaves
  every reliability measure unchanged: what it removes is as reproducible as what it keeps.
- The kept fraction is a subject trait: between subjects it varies ten times more than
  within a subject (ARC_L: SD 0.040 between, mean difference 0.004 within).  The share of
  a bundle's mass that leaks to non-dominant endpoints is itself a measurable quantity.
- CST_L failed with the largest-share rule, and a sagittal view showed why.  Its brainstem
  side has four patches: the ventral surface of the caudal pons and medulla (41% of the
  mass, absorbed into grey matter), the inferior edge of the white-matter mask (41%,
  absorbed at the brain-mask cut), the anterior pons where the corticopontine fibres end in
  the pontine nuclei (8%), and a posterior pontine exit (8%).  The two large patches are
  the same descending fibres, split by where the mask ends; which one is larger flips
  between sessions.  Taking any exit below the pons as the valid endpoint (patches 1 + 2)
  raised the kept fraction to 0.47 (0.23-0.64), the ICC to 0.958, and brought the
  within-subject difference back to the raw 4.3%; the remaining variability is the
  session-to-session change of the pontine leak, which is fODF-driven.  The kept fraction
  is the product of the two sides' shares (cortical dominant share 0.57, r = 0.99 between
  sessions; brainstem caudal share 0.82, r = 0.76).

## 6. What did not work

Rules that penalise the *shape* of the walk remove far more than they would on streamlines,
because every MAT walk is a full-probability path and the halo of a bundle is made of
walks that wander, not of a few outliers.

- **Flux pruning.**  Dropping voxels below an absolute fraction of the flux (0.1% of the
  streamlines) and re-solving cascades through the fanning terminations: 5-9% of the
  support voxels and 32-49% of the paths survive.  A local-contrast rule (flux below a
  fraction of the maximum within 5 mm) is gentler but still steep: 1% keeps about 94% of the
  paths, 3% about 70%, 10% between 21% and 35%.
- **Trunk tubes.**  Forbidding walks outside a tube around the bundle's own lobe (radius
  from the cross-section area holding 90% of the mass, plus a margin), only between the
  branch points of the fans, keeps 34-61% of the paths at +2 mm and 43-69% at +4 mm; a soft
  Gaussian tilt instead of a hard wall changes little.
- **Widest-path decomposition.**  Decomposing the flux into its widest routes (maximin
  paths on the voxel lattice, the fans as source and sink) is diffuse: the fifteen widest
  routes carry 5-16% of the flux, 0.5-1.5% each.  Modes of travel need a coarser graph than
  the voxel lattice.

The recommendation that follows: clean MAT bundles by where their walks end (committors,
endpoint patches) and account for their mass with the flux, not by taboos on trajectory
shape.

## 7. API

```python
from mcat import Chain, ChainConfig, prepare_launches, segment_bundles, restrict_endpoints
from mcat.fields import endpoint_patches, flux_orientation

res = segment_bundles(chain, specs, launches, keep_fields=True)["Left Corticospinal"]
F = res.fields                       # mcat.fields.BundleFields
n = F.visits()                       # (n_voxels,) = res.density_t - step 0 at the seeds
f26 = F.edge_flux()                  # (n_voxels, 26), geometry.OFFSETS order
F.conservation(f26)                  # {"in": ..., "out": ...} relative residuals
dirs, coherence = flux_orientation(f26, zooms)
phi = F.endpoint_sides([spec["start"], spec["end"]])   # (n_voxels, 2) visits by where they end
labels, shares = endpoint_patches(chain.to_volume(res.endpoints_a, "absorbing") * end_side)
filtered = segment_bundles(chain, [restrict_endpoints(spec, end=labels == 1)], launches)
```

The fields are for the plain chain; with spatial moments (`ChainConfig(moments=True)`) the
transitions are 4 x 4 blocks and `keep_fields` is refused.

## 8. Checks of this implementation

On one ds008825 session (3.4 million states, 2 mm grid, chain built in 18 s, 14 threads),
with the pyAFQ ROIs of the left arcuate and left corticospinal tract, solve tolerance 1e-8:

| | ARC_L | CST_L |
|---|---|---|
| expected streamlines | 258.6 | 98.2 |
| columns | 12 | 8 |
| largest voxel difference, fields vs `density_t`, relative to the count | 3e-17 | 4e-17 |
| conservation residual in / out | 3e-11 / 1e-10 | 7e-10 / 2e-10 |
| flux across one cut, over the count | 1.22 (coronal plane through the arc) | 1.05 (axial plane between the ROIs) |
| `edge_flux` | 0.1 s | 0.1 s |
| committors to the two ends | 37 s, 24 columns | 40 s, 16 columns |
| endpoint filter, caudal brainstem patches | | kept 0.52 (prototype chain: 0.55) |

The coronal cut through the arcuate counts its re-crossings (the arc bends through the
plane); the axial cut through the corticospinal tract is crossed once by nearly every walk.
The phantom tests (`tests/test_fields.py`) check the same identities on a straight bundle,
where the cut flux equals the streamline count to 1e-12.

## 9. Fixels

Fixel-based analysis measures each fibre population of a voxel (a lobe of the fODF, a
*fixel*) and needs to know which fixels a tract runs through.  From tractography that is
the number of streamlines assigned to each fixel.  MCAT gives the same number with no
streamlines, because its states carry a heading: the density is known per direction class,
and within a class the headings are spread as the chain's own transitions assume
(`geometry.vertex_visits`, the same weights that make the visitation ODF).  Summing the
expected visits over a lobe's directions gives the expected number of bundle-streamline
steps in that fixel; every step crosses into a neighbouring voxel, so this is the expected
number of bundle streamlines passing through the fixel.  The fixels of a voxel partition its
density exactly.

**Fixels** (`fixels.segment_lobes`): each direction of the chain's sphere climbs the fODF to
its local maximum over the sphere's triangulation, as MRtrix's fod2fixel segments lobes;
antipodal halves are one fixel; lobes whose saddle is at least 0.8 x the lower peak are one
fixel (the lmax-8 fit rings, and splits a lobe's shoulder off); lobes whose peak is under
0.1 x the voxel's maximum are not fixels, and their directions join the nearest fixel, so
every direction with a positive fODF belongs to one.  Fixels come out in MRtrix's layout
(count and offset per voxel, directions, one value per fixel) and are written as a fixel
directory by `fixels.write_fixel_directory` (.mif files written by hand; not yet checked
against MRtrix's reader).  The fixel directions are those of the chain's fODF, converted to world axes.

**Values** (`fixels.fixel_streamlines`): per fixel, the bundle's expected streamlines
through it.  With one more solve of the unconditioned chain (every walk from every seed, no
ROIs: `segment_bundles(chain, [dict(name="whole brain")], launches)`), the same number for
all traffic through the fixel, and the ratio is the bundle's share of the fixel, a soft fixel
segmentation; times the fixel's fibre density it is the bundle's part of that density.
`mcat.run(..., fixels=True)` (CLI `--fixels`) writes all three per bundle, and the bundle's
share of the solved bundles' streamlines.

On one ds008825 session (1.7 mm, 130 116 white-matter voxels): 257 134 fixels, 1.98 per
voxel (35% of voxels one, 43% two, 14% three, 5% four), segmented in 24 s; values for a
bundle in 27 s.  The CST's fixel values sum to 7179.4 against a density of 7180.9, with 1.6
in directions of no fixel.  In the 14 796 CST voxels with more than one fixel, the fixel
with the most CST holds a median 85% of the voxel's CST (10th percentile 55%): the
projection separates the CST from the fibres it crosses.  The share of all traffic is small
everywhere (at most 0.24; median 0.03 in the CST's densest fixels), because the
unconditioned chain counts every walk from every seed through the fixel, and in the
internal capsule most of those are not the pyAFQ-defined CST: thalamocortical and
corticopontine walks, walks ending in the deep grey matter, walks the include ROIs reject.
For a segmentation of the fixels among bundles, the share of the solved bundles' streamlines
is the better-conditioned quantity.

Two caveats.  Within a class the split between directions is a model assumption (the fODF
weighting the chain uses), not independent evidence: two fixels inside the same 35-degree
class cone are separated only by that weighting.  And the fixels must come from the fODF
the chain was built from, on its sphere; fixels from another segmentation can be matched by
direction but not assumed identical.

The exact version of this is a chain whose states are the fixels themselves, (voxel, lobe)
instead of (voxel, 26 classes), with transitions weighted by lobe-to-lobe compatibility
across the shared face: the bundle occupancy at a state is then the per-fixel density with
no projection, and the chain has about three states per voxel.

**Smoothing for fixel-based analysis.**  MRtrix smooths fixel data along a fixel-to-fixel
connectivity matrix built from a whole-brain tractogram (the fraction of streamlines through
fixel *i* that also pass fixel *j*).  The unconditioned chain holds the same information
without a tractogram: its edge flux between a fixel and the fixels of the neighbouring voxel
is the one-step connectivity, and the probability that a walk through fixel *i* also passes
fixel *j* a few voxels away is the corresponding multi-step transition (forwards, and
backwards for the part of the walk before *i*).  Writing that as MRtrix's sparse
connectivity format is a possible extension; it is not in this implementation.
