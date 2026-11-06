"""Seeds on the white/grey-matter interface, their half-walks along +/- each fODF peak, and the
seed probability of the arc bundle."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    chain = P.build_chain()
    L = P.launches(chain)
    V = chain.geo.vertices
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.6), gridspec_kw=dict(wspace=0.05))
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.02, right=0.93)
    P.show_plane(ax)
    sx, sy = L.seeds[:, 0], L.seeds[:, 1]
    ax.scatter(sx, sy, s=18, color=P.INK, zorder=6, lw=0)
    for p in range(L.pair_seed.size):
        x, y = L.seeds[L.pair_seed[p], :2]
        for d, c in zip(L.pair_d[p], (P.ORANGE, P.BLUE)):
            u = V[d, :2] * 1.3 * np.sqrt(L.pair_w[p])
            ax.annotate("", (x + u[0], y + u[1]), (x, y), zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, shrinkA=0, shrinkB=0,
                                        mutation_scale=9))
    ax.set_title("Seeds and their two half-walks (+p orange, −p blue)", loc="left", fontsize=12)
    P.label(ax, (19.5, 19.5), "one streamline per seed,\nsplit over its fODF peaks", color=P.INK2,
            fontsize=10, ha="center", va="center")

    res = P.segment(chain, P.arc_spec(include=False, exclude=False))
    sp = np.zeros((P.N, P.N))
    sp[sx, sy] = res.seed_prob
    P.show_plane(ax2)
    cm = P.prob_cmap()
    im = ax2.scatter(sx, sy, c=res.seed_prob, cmap=cm, norm=Normalize(0, 1), s=60, marker="s",
                     edgecolor=P.SURFACE, lw=0.6, zorder=6)
    r = P.arc_rois()
    P.outline(ax2, r["start"] | r["end"], P.ORANGE, lw=1.6)
    ax2.set_title("Arc: P(the streamline from this seed is in the bundle)", loc="left", fontsize=12)
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.01, shrink=0.7)
    cb.outline.set_visible(False)
    cb.set_label("seed probability", color=P.INK2)
    P.label(ax2, (19.5, 3.3), "endpoint ROIs", color=P.ORANGE, fontsize=10, ha="center")
    return fig
