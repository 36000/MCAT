# MCAT: Markov chain absorbing tractography

MCAT computes white-matter bundles as exact expectations over infinitely many streamlines, with no sampling.

- **The chain.** Probabilistic tracking is treated as an absorbing Markov chain on (white-matter voxel, direction) states. The transitions are the analytic hop geometry of [MITTENS](https://github.com/mattcieslak/MITTENS), with the walker's incoming direction remembered between voxels.
- **Bundle selection.** Include, exclude and endpoint ROIs are applied with forward and backward passes over that chain. This is transition path theory's reactive density, with inclusion–exclusion for the include ROIs.
- **pyAFQ.** It supplies everything else: the fODF it would track on, the tissue PVE, the seeds (WM/GM interface), the bundle definitions, and their ROIs warped to the subject.

```python
from AFQ.api.participant import ParticipantAFQ      # or GroupAFQ: every participant in turn
import mcat

afq = ParticipantAFQ(dwi, bval, bvec, t1, output_dir)
mcat.run(afq, n_proc=8, max_mem="16GB")              # -> <output_dir>/mcat/
mcat.view(f"{output_dir}/mcat", t1=t1_masked, bundles=["LeftArcuate", "LeftCorticospinal"], glyphs=True)
```

The command line runs pyAFQ and MCAT on one participant:

```bash
mcat sub-01_dwi.nii.gz sub-01_dwi.bval sub-01_dwi.bvec sub-01_T1w.nii.gz out/ --n-proc 8 --max-mem 16GB
```

## Install

```bash
pip install maturin                                   # MITTENS v2 builds with Rust (https://rustup.rs)
pip install "mcat-tractography[viewer] @ git+<this repository>"
```

MITTENS v2 (`mittens-dmri`) is not on PyPI yet, so pip builds it from GitHub with your Rust toolchain.

## Options

| `mcat.run(...)` | CLI | default | |
|---|---|---|---|
| `n_proc` | `--n-proc` | required | threads |
| `max_mem` | `--max-mem` | required | memory budget. Bundles are packed into groups that share every pass over the transition matrix; the budget sets how many fit. |
| `moments` | `--moments` | off | spatial first moments: about 5× slower |
| `tol` | `--tol` | 1e-10 | remaining walker mass at which the solves stop |
| `angle` | `--angle` | 35 | maximum turn per one-voxel hop (degrees). |
| `double_odf` | `--single-odf` | doubleODF | `--single-odf` gives first-order hops (see [MITTENS](https://github.com/mattcieslak/MITTENS))|
| `flux` | `--flux` | off | also write each bundle's edge flux and through-flux maps ([docs/fields.md](docs/fields.md)); not with moments |
| `fixels` | `--fixels` | off | also write an MRtrix fixel directory: each bundle's expected streamlines through every fixel of the fODF, and its share of all traffic ([docs/fields.md](docs/fields.md) section 9); one extra whole-brain solve |
| `bundles` | `--bundles` | all in pyAFQ's bundle dictionary | pyAFQ names, e.g. "Left Arcuate"; output files drop the spaces (`bundle-LeftArcuate`) |
| `viz` | `--no-viz` | on | the collage and viewer pages below |

Any pyAFQ setting goes on the `ParticipantAFQ`; with the CLI, pass it as `--afq-kwargs`. That covers the fODF model (`tracking_params={"odf_model": ...}`), the PVE, custom bundle dictionaries including subject-space ROIs, and `segmentation_params={"dist_to_atlas": ...}`. `--afq-kwargs` takes JSON, or a Python expression that gives a dict, for settings that are Python objects. The expression sees `afd` (`AFQ.data.fetch`), `abd` (`AFQ.api.bundle_dict`), the `AFQ.definitions` image and mapping classes, `np` and `nib`. For example, an infant run:

```bash
mcat dwi.nii.gz dwi.bval dwi.bvec T2w.nii.gz out/ --n-proc 12 --max-mem 16GB \
    --afq-kwargs 'dict(reg_template_spec=afd.read_pediatric_templates()["UNCNeo-withCerebellum-for-babyAFQ"],
                       bundle_info=abd.baby_bd(), pve="babyseg", brain_mask_definition=BabyBrainMask())'
```

**Solves that crawl** switch to a Krylov solver automatically, with no setting. A normal bundle loses about 10% of its walker mass per step, so 1e-10 takes 300–500 steps and Krylov never triggers. When nearly everything is walkable (ie, all-WM), walkers circle for thousands of steps. Once power iteration would need 500 or more further steps, or stalls, the rest of the series is solved with BiCGSTAB.

## Outputs (`<output_dir>/mcat/`, or `mcat-moments/`)

| file | |
|---|---|
| `<sub>_bundle-<Name>_desc-density_map.nii.gz` | expected steps of bundle streamlines in each voxel, summed over all seeds |
| `..._desc-endpoints_map.nii.gz` | expected bundle endpoints per grey-matter or edge voxel (two per streamline) |
| `..._desc-visitodf_dwimap.nii.gz` | visitation ODF: the directions the bundle's streamlines travel in each voxel, as SH in the fODF's basis (DIPY descoteaux07, lmax 8), in expected visits per steradian; it integrates to the density. Written where the density is at least 1e-3 of its peak |
| `..._desc-seedprob_probmap.nii.gz` | probability that the streamline seeded in each interface voxel belongs to the bundle |
| `..._desc-start_mask`, `_desc-end_mask` | the endpoint ROIs used (pyAFQ's `dist_to_atlas` applied) |
| `..._desc-core_path.tsv`, `_desc-core.tck` | the bundle's core: the cross-section ridge path, 100 nodes |
| `..._desc-throughflux_map.nii.gz`, `_desc-edgeflux_map.nii.gz` | with `flux`: expected bundle-streamline steps leaving each voxel, and the same per neighbour (26 volumes) |
| `<sub>_desc-mcat_fixels/` | with `fixels`: MRtrix fixel directory (`index`, `directions`, `<bundle>_streamlines`, `all_streamlines`, `<bundle>_share`, `<bundle>_share_of_bundles`) |
| `..._desc-summary.json` | settings, solve diagnostics, mass check |
| `<sub>_desc-mcat_table.tsv` | per bundle: expected streamlines, columns, steps, Krylov products, time |
| `<sub>_desc-mcat_dseg.nii.gz` | the tissue labels used: 1 WM, 2 GM, 3 CSF, 4 brain edge |
| `<sub>_desc-mcat_collage.png` | every bundle's density as sagittal, coronal and axial maximum-intensity projections, log scale (`viz`) |
| `<sub>_desc-mcat_view.html` | viewer page with every bundle: densities and cores (`viz`) |
| `..._bundle-<Name>_desc-view.html` | viewer page for one bundle, with its visitation-ODF glyphs and the fODF (`viz`) |

`mcat.view` returns an [ipyniivue](https://github.com/niivue/ipyniivue) widget. It shows the three slices and a 3-D render as an equal 2 × 2 grid, each bundle's density on a log scale from 1e-3 of its peak, and its core as a tube. Every volume is cropped to the head and the bundles, so the views fill their space, and nothing is transparent: the density is opaque over the background, and the 3-D view shows each bundle as an opaque surface (`isosurface`, on by default) inside a lit head cut open towards the camera. Double-click a view to enlarge it to the whole widget, and double-click again for all four (on the web page, Esc works too).

With `glyphs=True` (or a list of bundle names) it also draws the visitation ODFs as glyphs, like fODF glyphs. Where the fODF has a crossing, the glyph shows only the lobe the bundle uses. Glyphs follow the slices: every voxel at or above `glyph_floor` (1e-2 of the peak) gets one, drawn on the three slices through the crosshair, and redrawn as you click or scroll. The slices start at the core's midpoint. Options:

- `fodf=` takes an SH image on the same grid (pyAFQ's `*_model-csd_param-wm_dwimap.nii.gz`) and draws it as translucent grey glyphs at the same voxels, for comparison.
- `glyph_planes=(x, y, z)` starts the slices elsewhere (mm); `"all"` draws every glyph at once, which is heavy.
- `glyph_norm=False` sizes glyphs by visits instead of normalising each one.
- `glyph_color` is `"direction"` or `"bundle"`.

`mcat.viewer.save_html("bundles.html", folder, glyphs=True, ...)` writes the same scene as a standalone web page to open in any browser or share. It has controls for the layout, each bundle's density, core and glyphs, the opacities, and whether scrolling moves slices or zooms. Two log-scale sliders set the faintest density shown (down to 1e-6 of each bundle's peak) and the faintest voxel that gets a glyph (down to 1e-3). A checkbox (on by default) draws each bundle as an opaque surface at the density slider's level, redrawn as the slider moves; the notebook widget draws it at `floor`. The data are embedded and the page meshes the glyphs itself. NiiVue loads from a CDN; pass `niivue_js=` a local `niivue.umd.js` (npm `@niivue/niivue`, `dist/`) to embed it and work offline.

## Method

- **Tissue.** As pyAFQ's tracker: walkers move where the WM PVE exceeds 0.4 inside the fODF mask. Other voxels in the mask are grey matter (GM PVE at least the CSF PVE), which absorbs walkers, or CSF, which walkers never enter: every hop into CSF has zero probability, and the walker's other hops share its probability. A walker whose only ways on lead into CSF dies in white matter (invalid).
- **Geometry (MITTENS v2).**
  - A hop is one voxel long (of the smallest voxel side) and turns less than `angle`.
  - MITTENS enumerates every hop sequence inside a voxel, with the exact volume of start points, on DSI Studio's 642-direction sphere. Anisotropic voxels are exact.
  - MCAT adds each path's last direction class (the walker's memory) and its box of start points (for moments).
- **Transitions.**
  - A state is (voxel, one of 26 direction classes). Its transitions are sums over MITTENS paths of the fODF's turning probabilities times the doubleODF look-ahead, conditioned on the incoming direction.
  - Each transition is stored as a 1-byte slot into its class's template of (neighbour, class) pairs, plus a float64 weight. One copy serves forward (scatter) and backward (gather) steps.
- **Bundles.**
  - One backward solve per (avoid set × end kind) gives every state's probability of completing the bundle's requirements, shared by all seeds.
  - One forward solve from every seed's launch, weighted by its partner half's completion, gives the occupancy. Occupancy × completion is the bundle density.
- **Spatial moments.** States also carry their mean sub-voxel position, with a linear (Prather) closure that is exact per MITTENS path. Without them, every voxel restarts its paths from a uniform position, which makes bundles several times too wide. The closure can amplify around a few loops; those voxels fall back to the uniform closure.
- **Visitation ODFs.** The bundle density is kept per direction class (state), then spread over the sphere within each class as the chain itself assumes (q ∝ X·Y, the fODF times its cone sum). It is symmetrised and fitted with SH.
- **Fixels.** The walkers' headings are known per voxel, so a bundle's density splits exactly among the fODF's lobes: `mcat.fixels` segments the fixels and gives each bundle's expected streamlines through every fixel, and its share of all traffic, as an MRtrix fixel directory for fixel-based analysis without tractography ([docs/fields.md](docs/fields.md) section 9).
- **Bundle fields.** The occupancy and completion solutions behind the maps can be kept (`segment_bundles(keep_fields=True)`, `mcat.fields`): they give the bundle's edge flux (expected streamline steps along every transition, conserved, counting the streamlines across any cut), committors (where the walks through a voxel end) and an exact endpoint filter. [docs/fields.md](docs/fields.md) explains them and reports a test-retest evaluation.
- **Cores.** A first most-probable path finds the corridor. Each voxel joins the cross-section of its nearest path voxel. A second shortest path, with cost −log(density / its cross-section's maximum), follows the ridge without a length penalty.

