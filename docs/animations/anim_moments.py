"""Spatial moments: walkers launched along an oblique fibre, with every voxel restarting its
paths from a uniform position (left) vs carrying their mean sub-voxel position (right)."""
import numpy as np

import phantom as P

NX, NY = 44, 24
ANGLE = 18.0           # degrees from the x axis: not aligned with the grid
SEED = (3, 6)


def _chains():
    from mittens.io import default_sphere

    from mpat import Chain, ChainConfig

    V = np.asarray(default_sphere(), float)
    inv = P._sh_fit(V)
    a = np.radians(ANGLE)
    v = np.array([np.cos(a), np.sin(a), 0.0])
    sh = np.zeros((NX, NY, 3, inv.shape[1]), np.float32)
    sh[:, :, P.Z] = np.exp(30 * ((V @ v) ** 2 - 1)) @ inv
    tr = np.zeros((NX, NY, 3), bool)
    tr[1:-1, 1:-1, P.Z] = True
    ab = np.zeros((NX, NY, 3), np.int8)
    ab[:, :, 0] = ab[:, :, 2] = 2              # CSF above and below: stay in the plane
    ab[[0, -1], :, P.Z] = 3
    ab[:, [0, -1], P.Z] = 3
    P.CACHE.mkdir(parents=True, exist_ok=True)
    return [Chain.build(sh, (P.ZOOM,) * 3, tr, ab, ChainConfig(moments=m),
                        cache=str(P.CACHE / f"oblique_m{int(m)}.npz")) for m in (False, True)]


def _run(c, steps=64):
    from mpat.geometry import OFFSETS

    nc = c.ncomp
    b = int(np.flatnonzero((OFFSETS == [1, 0, 0]).all(1))[0])
    s = int(c.tindex[c.lin(np.array([[SEED[0], SEED[1], P.Z]]))[0]]) * c.nb + b
    x = np.zeros((c.n_states * nc, 1))
    x[s * nc] = 1.0                            # unit mass, centred (Q = 0)
    tot, y = x.copy(), np.empty_like(x)
    out = []
    for _ in range(steps):
        c.mul_forward(x, y)
        x, y = y, x
        tot += x
        img = lambda z: c.to_volume(z[::nc, 0].reshape(-1, c.nb).sum(1))[..., P.Z]  # noqa: E731
        out.append((img(x), img(tot)))
    return out


def make():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    runs = [_run(c) for c in _chains()]
    T = len(runs[0])
    xcut = 30
    a = np.radians(ANGLE)
    peak = max(r[-1][1].max() for r in runs)
    n_occ = LogNorm(peak * 1e-3, peak)
    cm = P.mass_cmap()

    fig = plt.figure(figsize=(12.6, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1], hspace=0.22, wspace=0.12, top=0.88, bottom=0.1,
                          left=0.03, right=0.97)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    ap = fig.add_subplot(gs[:, 1])
    ims = []
    xs = np.array([SEED[0], NX - 2])
    for ax, name in zip(axs, ("uniform restart in every voxel (moments off)",
                              "mean sub-voxel position carried (moments on)")):
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((0.5, 0.5), NX - 2, NY - 2, fc="white", ec=P.MUTED, lw=0.8, zorder=-1))
        ims.append(ax.imshow(np.zeros((NY, NX)), origin="lower", cmap=cm, norm=n_occ, interpolation="nearest"))
        ax.plot(xs, SEED[1] + (xs - SEED[0]) * np.tan(a), color=P.INK, lw=1, ls="--", alpha=0.6)
        ax.axvline(xcut, color=P.BLUE, lw=1.4, alpha=0.8)
        ax.set_xlim(-0.5, NX - 0.5)
        ax.set_ylim(-0.5, NY - 0.5)
        ax.set_title(name, fontsize=11, loc="left")
    P.label(axs[0], (xcut + 0.5, 1.2), "cross-section", color=P.BLUE, fontsize=9)
    P.label(axs[0], (8, SEED[1] + 5 * np.tan(a) - 3.2), "dashed: the fibre through the seed", color=P.INK2,
            fontsize=9)

    yy = np.arange(NY)
    ap.set_title("cross-section at the blue line", fontsize=11)
    ap.axvline(SEED[1] + (xcut - SEED[0]) * np.tan(a), color=P.INK, ls="--", lw=1, alpha=0.6)
    lines = [ap.plot([], [], color=c, lw=2, drawstyle="steps-mid")[0] for c in (P.MUTED, P.ORANGE)]
    ap.set_xlim(4, NY - 1)
    ap.set_xlabel("y (voxels)")
    ap.set_ylabel("expected visits")
    ymax = max(r[-1][1][xcut].max() for r in runs) * 1.1
    ap.set_ylim(0, ymax)
    lbl = [P.label(ap, (10, 0.01), t, color=c, fontsize=10, annotation_clip=False) for t, c in (("moments off", P.INK2),
                                                                      ("moments on", P.ORANGE))]
    sdt = ap.text(0.98, 0.97, "", transform=ap.transAxes, ha="right", va="top", fontsize=10, color=P.INK2)
    title = fig.text(0.03, 0.94, "", fontsize=13)

    frames = list(range(T)) + [T - 1] * 24

    def update(i):
        k = frames[i]
        sds = []
        for im, run, ln, lb in zip(ims, runs, lines, lbl):
            tr, tot = run[k]
            im.set_data(np.ma.masked_less(np.maximum(tot, 0) + 30 * np.maximum(tr, 0), n_occ.vmin).T)
            col = np.maximum(tot[xcut], 0)
            ln.set_data(yy, col)
            if col.sum() > 1e-9:
                m = (col * yy).sum() / col.sum()
                sds.append(np.sqrt((col * (yy - m) ** 2).sum() / col.sum()))
                j = int(np.argmax(col))
                lb.set_position((yy[j] + (1.2 if lb is lbl[1] else 2.0), col[j] * (0.9 if lb is lbl[1] else 0.5)))
                lb.set_alpha(1)
            else:
                lb.set_alpha(0)
        sdt.set_text(f"width (SD): {sds[0]:.2f} vs {sds[1]:.2f} voxels" if len(sds) == 2 else "")
        title.set_text(f"Step {k + 1}: one walker's mass, launched along an oblique fibre (18° to the grid)")

    return dict(fig=fig, update=update, frames=len(frames), fps=12, poster=len(frames) - 1)
