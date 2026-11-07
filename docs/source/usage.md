# Usage

## From Python

MCAT runs on top of a pyAFQ
[`ParticipantAFQ`](https://tractometry.org/pyAFQ/reference/api.html), or every participant of a
`GroupAFQ`. pyAFQ computes the derivatives MCAT needs: the fODF, the tissue PVE and brain
mask, the white/grey-matter interface, and the bundle ROIs warped to the subject. MCAT then
replaces the tracking and segmentation:

```python
from AFQ.api.participant import ParticipantAFQ
import mcat

afq = ParticipantAFQ(dwi, bval, bvec, t1, output_dir)
results = mcat.run(afq, n_proc=8, max_mem="16GB")     # -> <output_dir>/mcat/
```

`mcat.run` returns `{bundle name: BundleResult}`, or one such dict per participant for a
`GroupAFQ`, and writes the files described in {doc}`outputs`. The options are listed in
{doc}`options`.

Any pyAFQ setting goes on the `ParticipantAFQ`, for example:
- the fODF model (`tracking_params={"odf_model": ...}`)
- the PVE
- custom bundle dictionaries, including subject-space ROIs
- `segmentation_params={"dist_to_atlas": ...}`

```{note}
MCAT uses pyAFQ's ROI-based bundle definitions: include, exclude, start and end ROIs, and
whether a bundle may cross the midline. pyAFQ's streamline-based criteria (curvature, length,
cleaning) have no counterpart in an expectation and are not applied.
```

## From the command line

`mcat` runs pyAFQ and MCAT on one participant:

```bash
mcat sub-01_dwi.nii.gz sub-01_dwi.bval sub-01_dwi.bvec sub-01_T1w.nii.gz out/ --n-proc 8 --max-mem 16GB
```

The T1w image must be registered to the DWI (it is pyAFQ's `t1_file`). If no `odf_model` is
given, the command line uses CSD. Extra `ParticipantAFQ` arguments go in `--afq-kwargs`, as
JSON or as a Python expression that gives a dict. The expression can use:
- `afd` (`AFQ.data.fetch`) and `abd` (`AFQ.api.bundle_dict`)
- the `AFQ.definitions` image and mapping classes
- `np` and `nib`

For example, an infant run:

```bash
mcat dwi.nii.gz dwi.bval dwi.bvec T2w.nii.gz out/ --n-proc 12 --max-mem 16GB \
    --afq-kwargs 'dict(reg_template_spec=afd.read_pediatric_templates()["UNCNeo-withCerebellum-for-babyAFQ"],
                       bundle_info=abd.baby_bd(), pve="babyseg", brain_mask_definition=BabyBrainMask())'
```

All flags are listed in {doc}`cli`.

## Lower level: your own volumes

`mcat.run` is a thin layer over three calls that work on plain arrays. Use them to run MCAT
without pyAFQ, or on a phantom (as the figures in {doc}`algorithm` do):

```python
from mcat import Chain, ChainConfig, prepare_launches, segment_bundles

chain = Chain.build(sh, zooms, transient, absorb_class, ChainConfig(angle=35, double_odf=True))
launches = prepare_launches(chain, seeds)            # (S, 3) voxel indices
results = segment_bundles(chain, [dict(name="Arc", start=start_mask, end=end_mask,
                                       include=[roi_a, roi_b], exclude=avoid)],
                          launches, tol=1e-10, max_mem=8e9)
density = chain.to_volume(results["Arc"].density_t)
```

- `sh`: SH coefficients (DIPY `descoteaux07`, symmetric or full basis).
- `transient`: the white-matter mask.
- `absorb_class`: 0 outside, 1 grey matter, 2 CSF, 3 brain edge, wherever the voxel is not
  white matter.
