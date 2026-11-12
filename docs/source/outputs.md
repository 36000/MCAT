# Outputs

`mpat.run` writes to `<output_dir>/mpat/`, or to `mpat-moments/` with `moments=True`, next to
pyAFQ's own outputs.

| file | |
|---|---|
| `<sub>_bundle-<Name>_desc-density_map.nii.gz` | expected steps of bundle streamlines in each voxel, summed over all seeds |
| `..._desc-endpoints_map.nii.gz` | expected bundle endpoints per grey-matter or edge voxel (two per streamline) |
| `..._desc-visitodf_dwimap.nii.gz` | visitation ODF: the directions the bundle's streamlines travel in each voxel, as SH in the fODF's basis (DIPY descoteaux07, lmax 8), in expected visits per steradian; it integrates to the density. Written where the density is at least 1e-3 of its peak |
| `..._desc-seedprob_probmap.nii.gz` | probability that the streamline seeded in each interface voxel belongs to the bundle |
| `..._desc-start_mask`, `_desc-end_mask` | the endpoint ROIs used (pyAFQ's `dist_to_atlas` applied) |
| `..._desc-core_path.tsv`, `_desc-core.tck` | the bundle's core: the cross-section ridge path, 100 nodes |
| `..._desc-summary.json` | settings, solve diagnostics, mass check |
| `<sub>_desc-mpat_table.tsv` | per bundle: expected streamlines, columns, steps, Krylov products, time |
| `<sub>_desc-mpat_dseg.nii.gz` | the tissue labels used: 1 WM, 2 GM, 3 CSF, 4 brain edge |
| `<sub>_desc-chain_cache.npz` | the transition matrix, reused when the inputs and settings match |

## Reading the summary

Each bundle's `*_desc-summary.json` holds:

- `expected_bundle_streamlines`: the sum of the seed probabilities. Every seed contributes
  one streamline.
- `endpoint_mass_check`: the total endpoint mass divided by twice the expected streamlines.
  It should be 1.
- `backward` and `forward`: iterations, final residual, and, if the Krylov tail ran, where it
  started and how many products it took.
- `negative_mass_before_clip`: nonzero only with moments, whose linear closure can produce
  small negative masses (see {ref}`moments`).
- `solved_with`: the other bundles that shared this bundle's passes.

## How the maps are defined

The method behind each map is in {doc}`algorithm`, {ref}`outputs-derived`:
- the density is the reactive density summed over direction classes
- the visitation ODFs keep the direction classes
- the core is the ridge path of the density
