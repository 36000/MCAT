"""Tissue classes of the phantom, its fibres, and the 26 direction classes of a state."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt

    from mpat.geometry import OFFSETS, build_geometry

    d = P.tissue()
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_axes([0.0, 0.02, 0.5, 0.86])
    P.show_plane(ax)
    for (i, j), fib in P.fibre_field().items():
        if d[i, j] == 1:
            for v, w in fib:
                ax.plot([i - 0.38 * v[0], i + 0.38 * v[0]], [j - 0.38 * v[1], j + 0.38 * v[1]],
                        color=P.INK2, lw=0.7, solid_capstyle="round")
    lab = dict(fontsize=10, ha="center", va="center")
    P.label(ax, (19.5, 36.0), "edge of the brain mask: absorbs (valid end)", color=P.INK, **lab)
    P.label(ax, (19.5, 33.4), "grey matter: absorbs (valid end)", color=P.INK, **lab)
    P.label(ax, (11.5, 21.0), "CSF: never entered", color=P.BLUE, **lab)
    P.label(ax, (19.5, 9.3), "white matter: walkers move\n(sticks: fibre directions)", color=P.INK, **lab)
    P.label(ax, (8.0, 13.5), "arc", color=P.ORANGE, fontsize=11, fontweight="bold")
    P.label(ax, (20.6, 27.0), "stem", color=P.ORANGE, fontsize=11, fontweight="bold")
    ax.set_title("The documentation's phantom: one slice of a 2 mm brain", loc="left", fontsize=12)

    a3 = fig.add_axes([0.52, 0.0, 0.46, 0.78], projection="3d")
    g = build_geometry((P.ZOOM,) * 3, 35.0)
    V = g.vertices
    b = int(np.flatnonzero((OFFSETS == [1, 1, 0]).all(1))[0])
    sel = g.vbin == b
    a3.scatter(*V[~sel].T, s=4, color=P.MUTED, alpha=0.35, depthshade=False)
    a3.scatter(*V[sel].T, s=12, color=P.ORANGE, depthshade=False)
    for k, o in enumerate(OFFSETS):
        u = o / np.linalg.norm(o) * 1.35
        a3.plot([0, u[0]], [0, u[1]], [0, u[2]], color=P.ORANGE if k == b else P.INK2,
                lw=2.4 if k == b else 0.8)
    a3.set_box_aspect((1, 1, 1))
    a3.set_axis_off()
    a3.view_init(elev=22, azim=-60)
    fig.text(0.53, 0.905, "A state: (voxel, one of 26 direction classes)", fontsize=12)
    fig.text(0.53, 0.8, "lines: the 26 neighbour directions; dots: the 642 sphere directions,\n"
             "orange: the class of directions nearest neighbour (1, 1, 0)", fontsize=10, color=P.INK2)
    return fig
