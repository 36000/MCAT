"""A small synthetic subject that pyAFQ can process offline: two crossing white-matter
bundles in grey matter (simulated multi-shell-free DWI, b = 1000), a T1, a PVE, a brain mask,
and subject-space ROIs for each bundle (two include planes, start and end caps)."""
from pathlib import Path

import nibabel as nib
import numpy as np


def make_subject(root, shape=(22, 22, 10), vox=2.0, seed=0):
    from dipy.core.gradients import gradient_table
    from dipy.data import get_sphere
    from dipy.sims.voxel import multi_tensor

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = [-vox * shape[0] / 2 + 1, -vox * shape[1] / 2, -vox * shape[2] / 2]  # x < 0: left
    I, J, K = np.indices(shape)
    zc = (K >= 3) & (K <= 6)
    A = zc & (J >= 6) & (J <= 9) & (I >= 2) & (I <= shape[0] - 3)       # along x
    B = zc & (I >= 12) & (I <= 15) & (J >= 2) & (J <= shape[1] - 3)     # along y
    wm = A | B
    brain = (I >= 1) & (I <= shape[0] - 2) & (J >= 1) & (J <= shape[1] - 2) & (K >= 1) & (K <= shape[2] - 2)
    dirs = get_sphere(name="repulsion100").vertices[:60]
    bvals = np.r_[0, 0, np.full(60, 1000.0)]
    bvecs = np.r_[np.zeros((2, 3)), dirs]
    gtab = gradient_table(bvals, bvecs=bvecs)
    ev = np.array([1.7e-3, 0.2e-3, 0.2e-3])
    dwi = np.zeros(shape + (len(bvals),), np.float32)
    for v in np.argwhere(brain):
        a, b = A[tuple(v)], B[tuple(v)]
        if a and b:
            s, _ = multi_tensor(gtab, np.array([ev, ev]), S0=100, angles=[(90, 0), (90, 90)],
                                    fractions=[50, 50], snr=None)
        elif a or b:
            s, _ = multi_tensor(gtab, np.array([ev]), S0=100, angles=[(90, 0 if a else 90)], fractions=[100],
                                    snr=None)
        else:
            s, _ = multi_tensor(gtab, np.array([[0.8e-3] * 3]), S0=100, angles=[(0, 0)], fractions=[100],
                                    snr=None)
        dwi[tuple(v)] = s + rng.normal(0, 1.0, s.shape)
    files = {}

    def save(name, arr, dtype=np.float32):
        f = root / name
        nib.save(nib.Nifti1Image(np.asarray(arr, dtype), aff), f)
        return str(f)

    files["dwi"] = save("sub-01_dwi.nii.gz", dwi)
    np.savetxt(root / "sub-01_dwi.bval", bvals[None], fmt="%g")
    np.savetxt(root / "sub-01_dwi.bvec", bvecs.T, fmt="%.6f")
    files["bval"], files["bvec"] = str(root / "sub-01_dwi.bval"), str(root / "sub-01_dwi.bvec")
    # anatomy on a 2x finer grid, as a T1 usually is: the PVE then has to be resampled to the
    # DWI grid (pyAFQ's tracker resamples it linearly; so does MAT)
    up = lambda a: np.repeat(np.repeat(np.repeat(a, 2, 0), 2, 1), 2, 2)  # noqa: E731
    aff_t1 = np.diag([vox / 2, vox / 2, vox / 2, 1.0])
    aff_t1[:3, 3] = aff[:3, 3] - vox / 4

    def save_t1(name, arr, dtype=np.float32):
        f = root / name
        nib.save(nib.Nifti1Image(up(np.asarray(arr, dtype)), aff_t1), f)
        return str(f)

    files["t1"] = save_t1("sub-01_T1w.nii.gz", np.where(wm, 1.0, np.where(brain, 0.6, 0.0)))
    pve = np.zeros(shape + (3,), np.float32)
    pve[..., 2] = wm
    pve[..., 1] = brain & ~wm
    files["pves"] = [save_t1(f"sub-01_{t}_probseg.nii.gz", pve[..., i]) for i, t in enumerate(("csf", "gm", "wm"))]
    files["mask"] = save_t1("sub-01_brainmask.nii.gz", brain, np.uint8)
    rois = {}
    for name, axis, band in (("A", 0, A), ("B", 1, B)):
        lo, hi = np.argwhere(band)[:, axis].min(), np.argwhere(band)[:, axis].max()
        idx = I if axis == 0 else J
        plane = lambda k: (idx == k) & zc & ((J >= 5) & (J <= 10) if axis == 0 else (I >= 11) & (I <= 16))  # noqa: E731,B023
        cap = lambda k: (idx == k) & zc & ((J >= 6) & (J <= 9) if axis == 0 else (I >= 12) & (I <= 15))    # noqa: E731,B023
        rois[name] = {"include": [save(f"roi-{name}inc1.nii.gz", plane(lo + 4), np.uint8),
                                  save(f"roi-{name}inc2.nii.gz", plane(hi - 4), np.uint8)],
                      "start": save(f"roi-{name}start.nii.gz", cap(lo - 1), np.uint8),
                      "end": save(f"roi-{name}end.nii.gz", cap(hi + 1), np.uint8)}
    files["rois"] = rois
    files["wm"], files["A"], files["B"], files["affine"] = wm, A, B, aff
    return files


def make_afq(files, out):
    """A pyAFQ ParticipantAFQ that runs offline: given PVE and brain mask, identity mapping,
    the T1 as registration template, subject-space bundles, symmetric CSD (fast)."""
    from AFQ.api.bundle_dict import BundleDict
    from AFQ.api.participant import ParticipantAFQ
    from AFQ.definitions.image import ImageFile, PVEImages
    from AFQ.definitions.mapping import IdentityMap

    Path(out).mkdir(parents=True, exist_ok=True)
    bd = BundleDict({k: dict(v, space="subject", cross_midline=None) for k, v in files["rois"].items()},
                    resample_to=nib.load(files["t1"]), resample_subject_to=True)
    return ParticipantAFQ(files["dwi"], files["bval"], files["bvec"], files["t1"], str(out),
                          brain_mask_definition=ImageFile(path=files["mask"]),
                          # PVEImages, not PVEImage(path=...): the latter raises in pyAFQ 3.4
                          pve=PVEImages(*(ImageFile(path=p) for p in files["pves"])),
                          mapping_definition=IdentityMap(),
                          reg_template_spec=nib.load(files["t1"]), bundle_info=bd,
                          tracking_params={"odf_model": "CSD"}, csd_sh_order_max=6)
