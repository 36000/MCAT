"""MAT on top of pyAFQ.  Everything except the Markov chain comes from a pyAFQ
``ParticipantAFQ`` (or each participant of a ``GroupAFQ``): the fODF pyAFQ tracks on, the
tissue PVE and brain mask, the white/grey-matter interface (seeds), the bundle
definitions, and their ROIs warped to the subject.  Outputs go to ``<output_dir>/mat``
(``mat-moments`` with spatial moments), next to pyAFQ's own.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from .chain import ABSORB_CLASSES, Chain, ChainConfig
from .geometry import visitation_odf
from .solve import prepare_launches, segment_bundles

log = logging.getLogger("mat")
WM_THRESHOLD = 0.4   # pyAFQ's tracker stops where WM PVE <= gm_threshold (0.4)
ODF_FLOOR = 1e-3     # visitation ODFs are written where the density reaches this x its peak


def _on_grid(img, ref, order=1):
    """img on ref's grid: linear as pyAFQ's tracker does (dipy.align.resample), or nearest
    neighbour (order=0) for masks."""
    from dipy.align import resample
    from nibabel.processing import resample_from_to

    data = np.asarray(img.dataobj, np.float32)
    if img.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine, atol=1e-3):
        return data
    if order == 0:
        return np.asarray(resample_from_to(nib.Nifti1Image(data, img.affine), (ref.shape[:3], ref.affine),
                                           order=0).dataobj, np.float32)
    vols = [data] if data.ndim == 3 else [data[..., i] for i in range(data.shape[-1])]
    out = [np.asarray(resample(v, np.zeros(ref.shape[:3], np.float32), moving_affine=img.affine,
                               static_affine=ref.affine).get_fdata(), np.float32) for v in vols]
    return out[0] if data.ndim == 3 else np.stack(out, -1)


def tissue_labels(sh_img, pve_img, brain_mask_img=None):
    """As pyAFQ's tracker: walkers move where WM PVE > 0.4 inside the fODF mask; the mask's
    inner boundary counts as grey matter ("edge", so fibres leaving the field end validly);
    other in-mask voxels absorb as GM (GM PVE >= CSF PVE) or CSF.
    Returns transient, absorbing class (0 outside, 1 GM, 2 CSF, 3 edge), dseg (1 WM, 2 GM,
    3 CSF, 4 edge)."""
    from skimage.segmentation import find_boundaries

    mask = np.any(np.asarray(sh_img.dataobj) != 0, axis=-1)
    if brain_mask_img is not None:
        mask &= _on_grid(brain_mask_img, sh_img) > 0.5
    csf, gm, wm = np.moveaxis(_on_grid(pve_img, sh_img), -1, 0)
    edge = find_boundaries(mask, mode="inner") & mask
    transient = mask & ~edge & (wm > WM_THRESHOLD)
    absorb = np.zeros(mask.shape, np.int8)
    absorb[mask & ~transient & (gm >= csf)] = 1
    absorb[mask & ~transient & (gm < csf)] = 2
    absorb[edge] = 3
    dseg = np.where(transient, 1, np.where(absorb > 0, absorb + 1, 0)).astype(np.uint8)
    return transient, absorb, dseg


def seal_26(mask):
    """Make a thin ROI impassable for 26-connected walks.  Streamlines cannot slip through a
    one-voxel sheet; walkers stepping diagonally can.  Adding the sheet's 6-neighbours closes
    those gaps."""
    from scipy import ndimage

    return ndimage.binary_dilation(mask, ndimage.generate_binary_structure(3, 1))


def bundle_specs(afq, ref_img, transient, names=None):
    """MAT bundle specs from pyAFQ's bundle dictionary: include / exclude ROIs (sealed), start /
    end ROIs grown by pyAFQ's ``dist_to_atlas``, and the other hemisphere excluded for bundles
    that may not cross the midline.  Other pyAFQ criteria (curvature, length, cleaning) are
    streamline-based and do not apply."""
    from scipy.ndimage import distance_transform_edt

    bd, mapping = afq.export("bundle_dict"), afq.export("mapping")
    dist = afq.export("segmentation_params").get("dist_to_atlas", 4)
    R = ref_img.affine[:3, :3]
    tol = int(dist / np.mean(np.diag(np.linalg.cholesky(R.T @ R))))   # pyAFQ's tolerance_mm_to_vox
    x = nib.affines.apply_affine(ref_img.affine, np.indices(transient.shape).reshape(3, -1).T)[:, 0]
    x = x.reshape(transient.shape)
    specs = []
    for name in (names or list(bd)):
        rois = bd.transform_rois(name, mapping, ref_img)
        arr = lambda img: _on_grid(img, ref_img, order=0) > 0  # noqa: E731
        spec = dict(name=name, include=[seal_26(arr(m)) for m in rois.get("include", [])])
        exc = [seal_26(arr(m)) for m in rois.get("exclude", [])]
        for k in ("start", "end"):
            if k in rois:
                spec[k] = distance_transform_edt(~arr(rois[k])) - 0.5 <= tol   # pyAFQ's endpoint test
        if bd[name].get("cross_midline") is False:
            own = np.concatenate([np.atleast_1d(x[m]) for m in spec["include"] +
                                  [spec[k] for k in ("start", "end") if k in spec]])
            exc.append(x > 0 if own.mean() < 0 else x < 0)
        spec["exclude"] = np.any(exc, axis=0) if exc else None
        specs.append(spec)
    return specs


def _bytes(x):
    if isinstance(x, str):
        m = re.fullmatch(r"\s*([\d.]+)\s*([kmgt]?)b?\s*", x.lower())
        if not m:
            raise ValueError(f"cannot read memory size {x!r} (e.g. '16GB')")
        return float(m.group(1)) * 1024 ** "kmgt".find(m.group(2)) * (1024 if m.group(2) else 1)
    return float(x)


def run(afq, n_proc, max_mem, bundles=None, moments=False, tol=1e-10, angle=35.0, double_odf=True,
        cores=True):
    """Segment pyAFQ's bundles with MAT.

    afq         ``ParticipantAFQ`` or ``GroupAFQ`` (every participant in turn)
    n_proc      threads
    max_mem     memory budget, bytes or e.g. "16GB": sets how many bundles share each solve
    bundles     pyAFQ bundle names (default: all in the bundle dictionary)
    moments     spatial first moments (narrower, more accurate bundles; ~5x slower)
    tol         remaining walker mass at which the solves stop (relative)
    angle       maximum turn per one-voxel hop (degrees)
    double_odf  weight each hop by the fODF where it lands (False: singleODF, like pyAFQ's
                first-order probabilistic tracker)
    cores       also write each bundle's ridge-path core (mat.core)
    Returns {bundle: BundleResult} (a dict per participant for a GroupAFQ)."""
    import numba

    if hasattr(afq, "pAFQ_list"):
        return {f"sub-{s}_ses-{t}": run(p, n_proc, max_mem, bundles, moments, tol, angle, double_odf, cores)
                for s, t, p in zip(afq.valid_sub_list, afq.valid_ses_list, afq.pAFQ_list)}
    numba.set_num_threads(max(1, min(int(n_proc), numba.config.NUMBA_NUM_THREADS)))
    t0 = time.time()
    sh_img = nib.load(afq.export(f"{afq.export('tracking_params')['odf_model'].lower()}_params"))
    transient, absorb, dseg = tissue_labels(sh_img, nib.load(afq.export("pve_internal")),
                                            nib.load(afq.export("brain_mask")))
    out = Path(afq.kwargs["output_dir"]) / ("mat-moments" if moments else "mat")
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / Path(afq.kwargs["base_fname"]).name)
    nib.save(nib.Nifti1Image(dseg, sh_img.affine), f"{stem}_desc-mat_dseg.nii.gz")
    specs = bundle_specs(afq, sh_img, transient, bundles)
    cfg = ChainConfig(angle=angle, double_odf=double_odf, moments=moments)
    sh = np.asarray(sh_img.dataobj, np.float32)
    chain = Chain.build(sh, sh_img.header.get_zooms()[:3], transient, absorb, cfg,
                        cache=f"{stem}_desc-chain_cache.npz")
    seeds = np.argwhere((_on_grid(nib.load(afq.export("wm_gm_interface")), sh_img, 0) > 0) & transient)
    launches = prepare_launches(chain, seeds)
    chain.release_pmfs()
    rows = []

    def write(res):
        name = re.sub(r"[^A-Za-z0-9]", "", res.name)
        b = f"{stem}_bundle-{name}"
        spec = next(s for s in specs if s["name"] == res.name)
        sp = np.zeros(chain.shape, np.float32)
        np.add.at(sp, tuple(res.seed_vox.T), res.seed_prob)
        maps = dict(seedprob=(sp, "probmap"), density=(chain.to_volume(res.density_t), "map"),
                    endpoints=(chain.to_volume(res.endpoints_a, "absorbing"), "map"))
        d = res.density_t
        keep = np.flatnonzero(d >= ODF_FLOOR * d.max()) if d.max() > 0 else np.zeros(0, np.int64)
        ijk = tuple((np.stack(np.unravel_index(chain.t_lin[keep], chain.pshape), 1) - 1).T)
        coef = visitation_odf(res.density_class[keep], sh[ijk], chain.geo)
        odf = np.zeros(chain.shape + coef.shape[1:], np.float32)
        odf[ijk] = coef
        maps["visitodf"] = (odf, "dwimap")
        for desc, (vol, suffix) in maps.items():
            nib.save(nib.Nifti1Image(vol, sh_img.affine), f"{b}_desc-{desc}_{suffix}.nii.gz")
        for k in ("start", "end"):
            if spec.get(k) is not None:
                nib.save(nib.Nifti1Image(spec[k].astype(np.uint8), sh_img.affine), f"{b}_desc-{k}_mask.nii.gz")
        with open(f"{b}_desc-summary.json", "w") as fh:
            json.dump(dict(bundle=res.name, config=cfg.key(), result=res.info,
                           valid_ends=[ABSORB_CLASSES[c] for c in (1, 3)],
                           density="expected steps of bundle streamlines in each voxel, all seeds",
                           endpoints="expected bundle endpoints per voxel (two per streamline)",
                           visitodf="the walkers' directions per voxel: SH (descoteaux07, legacy, "
                                    "lmax 8), expected visits per steradian",
                           seedprob="P(the streamline seeded here belongs to the bundle)"), fh, indent=1)
        i = res.info
        rows.append(dict(bundle=res.name, expected_streamlines=round(i["expected_bundle_streamlines"], 4),
                         columns=i["n_columns"], backward_steps=i["backward"]["iterations"],
                         forward_steps=i["forward"]["iterations"],
                         krylov_products=i["backward"].get("tail_products", 0) + i["forward"].get("tail_products", 0),
                         seconds=i["seconds"]))
        log.info("%s: %.2f expected streamlines", res.name, i["expected_bundle_streamlines"])

    results = segment_bundles(chain, specs, launches, tol=tol, max_mem=_bytes(max_mem), on_result=write)
    with open(f"{stem}_desc-mat_table.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    if cores:
        from .core import write_cores
        write_cores(out)
    log.info("MAT: %d bundles in %.0fs -> %s", len(results), time.time() - t0, out)
    return results
