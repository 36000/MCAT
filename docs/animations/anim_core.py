"""A bundle's core (mcat.core.bundle_core), stage by stage: anchors, the most-probable path,
cross-sections, and the ridge path."""
import numpy as np

import phantom as P


def _stages(density, endpoints, start, end, affine):
    """The intermediate results of mcat.core.bundle_core (same steps, kept for drawing)."""
    from scipy.ndimage import label
    from scipy.sparse.csgraph import dijkstra

    from mcat.core import _anchor, _graph, _path, split_ends

    d = np.clip(density, 0, None)
    ep = np.clip(endpoints, 0, None)
    dn = d / d.max()
    vox_mm = float(np.abs(np.linalg.det(affine[:3, :3])) ** (1 / 3))
    S_roi, E_roi, how = split_ends(ep, affine, start, end)
    keep = dn > 1e-6
    lab, _ = label(keep, np.ones((3, 3, 3)))
    keep = lab == lab[np.unravel_index(np.argmax(dn), dn.shape)]
    bod = keep & (dn >= 1e-2)
    s_vox, e_vox = _anchor(ep, S_roi, bod), _anchor(ep, E_roi, bod)
    cost1 = np.where(keep, -np.log(np.where(keep, dn, 1.0)), 0.0)
    G, idx, lin = _graph(cost1, keep, vox_mm)
    src, dst = np.array([lin[tuple(s_vox)]]), np.array([lin[tuple(e_vox)]])
    p0 = _path(G, src, dst)
    dist, _, owner = dijkstra(G, indices=p0, min_only=True, return_predecessors=True)
    dv = d[tuple(idx.T)]
    ok = np.isfinite(dist)
    mx = np.zeros(len(idx))
    np.maximum.at(mx, owner[ok], dv[ok])
    rel = np.zeros(len(idx))
    rel[ok] = dv[ok] / mx[owner[ok]]
    cost2 = np.zeros(d.shape)
    cost2[tuple(idx.T)] = np.where(ok, -np.log(np.maximum(rel, 1e-300)), 50.0)
    G2, idx2, _ = _graph(cost2, keep, vox_mm)
    p = _path(G2, src, dst)
    pos = {int(v): k for k, v in enumerate(p0)}
    own_img = np.full(d.shape, np.nan)
    own_img[tuple(idx[ok].T)] = [pos.get(int(o), np.nan) for o in owner[ok]]
    return dict(s=s_vox, e=e_vox, first=idx[p0], ridge=idx2[p], owner=own_img, rel=rel, how=how)


def make():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from mcat import bundle_core

    chain = P.build_chain()
    spec = P.arc_spec(include=False, exclude=False)
    res = P.segment(chain, spec)
    dens = chain.to_volume(res.density_t)
    ep = chain.to_volume(res.endpoints_a, "absorbing")
    aff = np.diag([P.ZOOM] * 3 + [1.0])
    st = _stages(dens, ep, spec["start"], spec["end"], aff)
    core = bundle_core(dens, aff, ep, spec["start"], spec["end"])
    cxy = core.xyz[:, :2] / P.ZOOM

    img = dens[..., P.Z]
    peak = img.max()
    norm = LogNorm(peak * 1e-3, peak)
    fig = plt.figure(figsize=(11.6, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06, top=0.86, bottom=0.03)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    P.show_plane(ax)
    ax.imshow(np.ma.masked_less(img, norm.vmin).T, origin="lower", cmap=P.mass_cmap(), norm=norm,
              interpolation="nearest")
    epi = ep[..., P.Z]
    ax.imshow(np.ma.masked_less_equal(epi, epi.max() * 1e-3).T, origin="lower", cmap=P.prob_cmap(),
              norm=LogNorm(epi.max() * 1e-3, epi.max()), interpolation="nearest")
    ax.set_title("bundle density (orange) and endpoints (blue)", fontsize=11)

    P.show_plane(ax2)
    own = st["owner"][..., P.Z]
    ax2.imshow(np.ma.masked_less(img, norm.vmin).T, origin="lower", cmap=P.mass_cmap(), norm=norm,
               interpolation="nearest", alpha=0.45)
    ribs = []                                   # voxel edges between different cross-sections
    for i in range(P.N - 1):
        for j in range(P.N - 1):
            a = own[i, j]
            if not np.isfinite(a):
                continue
            if np.isfinite(own[i + 1, j]) and own[i + 1, j] != a:
                ribs.append([(i + 0.5, j - 0.5), (i + 0.5, j + 0.5)])
            if np.isfinite(own[i, j + 1]) and own[i, j + 1] != a:
                ribs.append([(i - 0.5, j + 0.5), (i + 0.5, j + 0.5)])
    from matplotlib.collections import LineCollection
    sec = LineCollection(ribs, colors=P.INK2, linewidths=0.9, alpha=0.0, zorder=6)
    ax2.add_collection(sec)
    t2 = ax2.set_title("", fontsize=11)

    def poly(ax_, color, ls="-", lw=2.4, z=8):
        return ax_.plot([], [], color=color, ls=ls, lw=lw, zorder=z, solid_capstyle="round")[0]

    first = [poly(a, P.INK, "--", 2.0) for a in (ax, ax2)]
    ridge = [poly(a, P.BLUE, "-", 2.6, 9) for a in (ax, ax2)]
    final = ax.plot([], [], "o-", color=P.AQUA, ms=3.2, lw=1.6, zorder=10, mec=P.SURFACE, mew=0.6)[0]
    anchors = [a.scatter([], [], s=70, color=P.INK, zorder=11, marker="D", edgecolor=P.SURFACE, lw=1.5)
               for a in (ax, ax2)]
    legend = [("◆ anchors: endpoint-weighted centres of each end", P.INK),
              ("- - most-probable path (cuts the corner)", P.INK),
              ("—  ridge path: max of each cross-section", P.BLUE),
              ("●  core: smoothed, 100 nodes", P.AQUA)]
    leg = [fig.text(0.53, 0.115 - 0.028 * k, t, fontsize=9.5, color=c, alpha=0.0) for k, (t, c) in enumerate(legend)]
    title = fig.text(0.03, 0.94, "", fontsize=13)
    F, R = st["first"][:, :2], st["ridge"][:, :2]

    seq = ([("anchors", a) for a in np.linspace(0, 1, 12)] + [("first", a) for a in np.linspace(0, 1, 30)]
           + [("sections", a) for a in np.linspace(0, 1, 16)] + [("hold", 1)] * 8
           + [("ridge", a) for a in np.linspace(0, 1, 30)] + [("final", a) for a in np.linspace(0, 1, 12)]
           + [("end", 1)] * 24)
    order = ["anchors", "first", "sections", "hold", "ridge", "final", "end"]

    def upto(stage, name):
        return order.index(stage) > order.index(name)

    def update(i):
        stage, a = seq[i]
        for sc in anchors:
            sc.set_offsets([st["s"][:2], st["e"][:2]])
            sc.set_alpha(a if stage == "anchors" else 1.0)
        n = int(round(a * len(F))) if stage == "first" else (len(F) if upto(stage, "first") else 0)
        for ln in first:
            ln.set_data(F[:n, 0], F[:n, 1])
        sec.set_alpha(0.9 * (a if stage == "sections" else (1.0 if upto(stage, "sections") else 0.0)))
        m = int(round(a * len(R))) if stage == "ridge" else (len(R) if upto(stage, "ridge") else 0)
        for ln in ridge:
            ln.set_data(R[:m, 0], R[:m, 1])
        k = int(round(a * len(cxy))) if stage == "final" else (len(cxy) if stage == "end" else 0)
        final.set_data(cxy[:k, 0], cxy[:k, 1])
        shown = {"anchors": 1, "first": 2, "sections": 2, "hold": 2, "ridge": 3, "final": 4, "end": 4}[stage]
        for j, t in enumerate(leg):
            t.set_alpha(1.0 if j < shown else 0.0)
        title.set_text({
            "anchors": "1. Anchor each end at the endpoint-weighted centre of its endpoints",
            "first": "2. Most-probable path: Dijkstra with cost −log(density / peak) per mm",
            "sections": "3. Cross-sections: every voxel joins its nearest first-path voxel",
            "hold": "3. Cross-sections: every voxel joins its nearest first-path voxel",
            "ridge": "4. Ridge path: cost −log(density / its cross-section's max): free along the ridge",
            "final": "5. Smooth and resample: the bundle's core",
            "end": "5. Smooth and resample: the bundle's core"}[stage])
        t2.set_text("cross-sections (thin lines between them)" if upto(stage, "first") else "")

    return dict(fig=fig, update=update, frames=len(seq), fps=12, poster=len(seq) - 1)
