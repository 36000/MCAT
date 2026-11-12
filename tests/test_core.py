"""Bundle cores (core.py) and the NiiVue viewer helpers (niivue.py)."""
import nibabel as nib
import numpy as np
import pytest

from mpat.core import bundle_core, split_ends, write_core


def _arc_bundle(R=14.0, sigma=1.6, dip=0.25, shape=(44, 44, 11)):
    """Quarter-circle bundle (radius R voxels, Gaussian cross-section) whose ridge is weaker
    in the middle of the bend -- where a plain most-probable path cuts the corner -- with
    endpoint voxels just beyond both ends.  Returns density, endpoints, start, end, arc."""
    cx = cy = 4.0
    zc = shape[2] // 2
    I, J, K = np.indices(shape).astype(float)
    r = np.hypot(I - cx, J - cy)
    th = np.arctan2(J - cy, I - cx)
    inside = (th >= 0) & (th <= np.pi / 2)
    ridge = 1 - dip * np.sin(2 * np.clip(th, 0, np.pi / 2))      # weakest at 45 degrees
    d = np.where(inside, ridge * np.exp(-((r - R) ** 2 + (K - zc) ** 2) / (2 * sigma ** 2)), 0.0)
    ep = np.zeros(shape)
    start = np.zeros(shape, bool)
    end = np.zeros(shape, bool)
    for dz in (-1, 0, 1):
        for dr in (-1, 0, 1):
            start[int(cx + R + dr), int(cy) - 1, zc + dz] = True    # just below theta = 0
            end[int(cx) - 1, int(cy + R + dr), zc + dz] = True      # just left of theta = 90
    ep[start] = 1.0
    ep[end] = 1.0
    t = np.linspace(0, np.pi / 2, 400)
    arc = np.stack([cx + R * np.cos(t), cy + R * np.sin(t), np.full_like(t, zc)], 1)
    return d, ep, start, end, arc


def test_ridge_path_follows_the_bend():
    d, ep, start, end, arc = _arc_bundle()
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    c = bundle_core(d, aff, ep, start, end, n_nodes=60)
    assert c.ends == "start and end ROIs"
    ijk = nib.affines.apply_affine(np.linalg.inv(aff), c.xyz)
    dist = np.sqrt(((ijk[:, None] - arc[None]) ** 2).sum(-1)).min(1)   # voxels
    inner = slice(5, -5)                                                # away from the anchors
    assert dist[inner].mean() < 0.6 and dist[inner].max() < 1.2
    # the first (plain most-probable) path cuts the weak bend: shorter, off the arc
    f_ijk = nib.affines.apply_affine(np.linalg.inv(aff), c.first_xyz)
    f_dist = np.sqrt(((f_ijk[:, None] - arc[None]) ** 2).sum(-1)).min(1)
    print("ridge path max %.2f vox, first path max %.2f vox" % (dist[inner].max(), f_dist[inner].max()))
    assert c.info["first_path_length_mm"] < c.length_mm and f_dist[inner].max() > 2 * dist[inner].max()
    assert c.density[inner].min() > 0.6                                 # on the ridge, not its flank
    # oriented start -> end
    assert np.linalg.norm(ijk[0] - arc[0]) < np.linalg.norm(ijk[0] - arc[-1])


def test_split_ends_without_rois():
    shape = (30, 10, 10)
    aff = np.eye(4)
    aff[0, 3] = -15.0                                    # x < 0 for i < 15
    ep = np.zeros(shape)
    ep[2, 5, 5] = ep[27, 5, 5] = 1.0
    a, b, how = split_ends(ep, aff)
    assert how == "left vs right hemisphere" and a[2, 5, 5] and b[27, 5, 5]
    aff[0, 3] = 5.0                                      # all in one hemisphere: clusters
    a, b, how = split_ends(ep, aff)
    assert how == "two endpoint clusters" and a.sum() == 1 and b.sum() == 1 and not (a & b).any()
    roi = np.zeros(shape, bool)
    roi[27, 5, 5] = True
    a, b, how = split_ends(ep, aff, end=roi)              # as pyAFQ: the end ROI is the last end
    assert how == "all other endpoints vs end ROI" and a[2, 5, 5] and b[27, 5, 5]
    a, b, how = split_ends(ep, aff, start=roi)
    assert how == "start ROI vs all other endpoints" and a[27, 5, 5] and b[2, 5, 5]


def test_split_ends_by_include_order():
    """No start / end ROI: the end nearer the first include ROI comes first, as pyAFQ orients
    streamlines from its first include ROI to its last -- even across hemispheres."""
    shape = (30, 10, 10)
    aff = np.eye(4)
    aff[0, 3] = -15.0
    ep = np.zeros(shape)
    ep[2, 5, 5] = ep[27, 5, 5] = 1.0
    inc = np.zeros(shape, np.uint8)
    inc[24, 5, 5], inc[15, 5, 5], inc[5, 5, 5] = 1, 2, 3   # right, midline, left (pyAFQ's callosum)
    a, b, how = split_ends(ep, aff, include=inc)
    assert how == "first vs last include ROI" and a[27, 5, 5] and b[2, 5, 5]


def test_write_core(tmp_path):
    d, ep, start, end, _ = _arc_bundle()
    c = bundle_core(d, np.eye(4), ep, start, end, n_nodes=50)
    tsv, tck = write_core(c, str(tmp_path / "x_bundle-A"))
    rows = np.loadtxt(tsv, skiprows=1)
    assert rows.shape == (50, 5) and np.allclose(rows[:, 1:4], c.xyz, atol=1e-3)
    sl = nib.streamlines.load(tck).streamlines[0]
    assert np.allclose(sl, c.xyz, atol=1e-4)


def test_tube_and_viewer(tmp_path):
    from mpat.viewer import log_density_u8, tube_obj

    xyz = np.stack([np.linspace(0, 30, 20), np.sin(np.linspace(0, 3, 20)) * 5, np.zeros(20)], 1)
    obj = tube_obj(xyz, radius=1.5, sides=10).decode().splitlines()
    v = np.array([list(map(float, ln.split()[1:])) for ln in obj if ln.startswith("v ")])
    f = np.array([list(map(int, ln.split()[1:])) for ln in obj if ln.startswith("f ")])
    assert len(v) == 20 * 10 + 2 and len(f) == 2 * 19 * 10 + 2 * 10 and f.min() == 1 and f.max() == len(v)
    ring = v[:200].reshape(20, 10, 3)
    assert np.allclose(np.linalg.norm(ring - xyz[:, None], axis=-1), 1.5, atol=1e-3)
    u8 = log_density_u8(np.array([0.0, 1e-4, 1e-3, 1e-2, 1.0]), floor=1e-3)
    assert list(u8) == [0, 0, 1, 86, 255]          # 1 + 254 * (1/3)

    pytest.importorskip("ipyniivue")
    from mpat.viewer import view

    d, ep, start, end, _ = _arc_bundle()
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    pre = tmp_path / "sub-01_bundle-ARC"
    for k, a in (("density_map", d), ("endpoints_map", ep), ("start_mask", start), ("end_mask", end)):
        nib.save(nib.Nifti1Image(a.astype(np.float32), aff), f"{pre}_desc-{k}.nii.gz")
    nib.save(nib.Nifti1Image((d > 0).astype(np.int8), aff), str(tmp_path / "sub-01_desc-mpat_dseg.nii.gz"))
    nv = view(tmp_path)
    assert [v.name for v in nv.volumes][1:] == ["ARC_logdensity.nii.gz"]
    assert "mpat_dseg" in nv.volumes[0].name and [m.name for m in nv.meshes] == ["ARC_core.obj"]
    assert (tmp_path / "sub-01_bundle-ARC_desc-core.tck").exists()     # computed and saved


def test_glyph_mz3():
    import gzip
    import struct

    from dipy.core.sphere import Sphere
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    from mpat.viewer import glyph_mz3

    u = get_sphere(name="repulsion724").vertices
    _, inv = sh_to_sf_matrix(Sphere(xyz=u), sh_order_max=8, return_inv=True)
    coef = np.stack([np.abs(u @ d) ** 30 @ inv for d in ([1.0, 0, 0], [0, 0, 1.0])] + [np.zeros(45)])
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-10, 5, 0]
    ijk = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
    buf = gzip.decompress(glyph_mz3(coef, ijk, aff, scale=0.9))
    magic, attr, nf, nv, skip = struct.unpack("<HHIII", buf[:16])
    assert (magic, attr, skip) == (23117, 7, 0)
    faces = np.frombuffer(buf, "<u4", nf * 3, 16).reshape(-1, 3)
    verts = np.frombuffer(buf, "<f4", nv * 3, 16 + 12 * nf).reshape(-1, 3)
    rgba = np.frombuffer(buf, np.uint8, nv * 4, 16 + 12 * nf + 12 * nv).reshape(-1, 4)
    assert len(buf) == 16 + 12 * nf + 16 * nv and faces.max() == nv - 1
    centres = nib.affines.apply_affine(aff, ijk[:2])            # the empty ODF draws nothing
    near = np.argmin(np.linalg.norm(verts[:, None] - centres[None], axis=-1), axis=1)
    off = verts - centres[near]
    assert np.linalg.norm(off, axis=1).max() <= 0.9 + 1e-5      # half the voxel side x scale
    for g, axis in ((0, 0), (1, 2)):                            # lobes along x, then z
        o = off[near == g]
        tip = np.argmax(np.linalg.norm(o, axis=1))
        assert abs(o[tip, axis]) > 0.89 and rgba[near == g][tip, axis] == 255
