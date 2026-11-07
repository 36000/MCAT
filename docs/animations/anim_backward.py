"""Backward pass: the probability that a walker in each state completes the requirement (here,
ends in the arc's end ROI), filled in backwards from the ROI one step at a time."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    from mcat.geometry import OFFSETS

    chain = P.build_chain()
    rois = P.arc_rois()
    end_pad = np.pad(P.to3d(rois["end"]), 1).ravel()
    absw = end_pad[chain.a_lin].astype(float)[:, None]
    r = chain.absorb_rhs(absw)
    beta = r.copy()
    total = r.copy()
    hist = [total[:, 0].copy()]
    nxt = np.empty_like(r)
    term = r.copy()
    for _ in range(80):
        chain.mul_backward(term, nxt)
        term, nxt = nxt, term
        total += term
        hist.append(total[:, 0].copy())
        if term.max() < 1e-6:
            break
    beta = hist

    cm = P.prob_cmap()
    norm = Normalize(0, 1)
    fig = plt.figure(figsize=(11.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08, top=0.85, bottom=0.08)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    P.show_plane(ax1)
    im = ax1.imshow(np.zeros((P.N, P.N)), origin="lower", cmap=cm, norm=norm, interpolation="nearest")
    P.outline(ax1, rois["end"], P.ORANGE, lw=2)
    P.label(ax1, (27, 1.2), "end ROI", color=P.ORANGE, fontsize=10)
    ax1.set_title("best heading: max over direction classes")
    x0, x1, y0, y1 = 25, 33, 7, 15
    ax1.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec=P.INK, lw=1.2,
                            ls="--", zorder=8))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm, cm), ax=ax1, orientation="horizontal", fraction=0.04,
                      pad=0.02, shrink=0.7)
    cb.set_label("β: P(walker ends in the end ROI)", color=P.INK2)
    cb.outline.set_visible(False)

    P.show_plane(ax2)
    ax2.set_xlim(x0 - 0.5, x1 + 0.5)
    ax2.set_ylim(y0 - 0.5, y1 + 0.5)
    P.outline(ax2, rois["end"], P.ORANGE, lw=2)
    ax2.set_title("each direction class separately (spoke length = β)")
    inplane = np.flatnonzero(OFFSETS[:, 2] == 0)
    dirs = OFFSETS[inplane, :2] / np.linalg.norm(OFFSETS[inplane, :2], axis=1, keepdims=True)
    tv = [(i, j) for i in range(x0, x1 + 1) for j in range(y0, y1 + 1)
          if chain.tindex[chain.lin(np.array([[i, j, P.Z]]))[0]] >= 0]
    sidx = np.array([[P.state_index(chain, (i, j, P.Z), b) for b in inplane] for i, j in tv])
    lc = LineCollection([], linewidths=2.4, capstyle="round", zorder=6)
    ax2.add_collection(lc)
    ax2.scatter([x for x, _ in tv], [y for _, y in tv], s=16, color=P.INK, zorder=7, lw=0)
    P.label(ax2, (x0 - 0.3, y1 + 0.15), "dot: the voxel; spokes: headings, each as long as its β",
            color=P.INK2, fontsize=9)
    title = fig.text(0.04, 0.94, "", fontsize=13)

    frames = []
    for k in range(len(beta)):
        frames += [k] * (3 if k < 15 else 1)
    frames += [len(beta) - 1] * 18

    def update(i):
        k = frames[i]
        b = beta[k]
        img = P.per_voxel(chain, b, "max")
        im.set_data(np.ma.masked_less(img, 1e-3).T)
        segs, cols = [], []
        for (x, y), ss in zip(tv, sidx):
            for d, s in zip(dirs, ss):
                v = b[s]
                if v > 1e-3:
                    segs.append([(x, y), (x + 0.48 * v * d[0], y + 0.48 * v * d[1])])
                    cols.append(cm(0.35 + 0.65 * v))
        lc.set_segments(segs)
        lc.set_color(cols)
        title.set_text(f"Backward pass, {k} steps back: β = P(this state's walker ends in the end ROI)")

    return dict(fig=fig, update=update, frames=len(frames), fps=12, poster=len(frames) - 1)
