"""Visitation ODFs: the directions a bundle's walkers travel, vs the fODF, at the crossing."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt

    from mcat.geometry import _sh_basis, visitation_odf

    chain = P.build_chain()
    sh = P.volumes()[0]
    specs = {"arc": P.arc_spec(include=False, exclude=False, name="arc"),
             "stem": dict(name="stem", start=P.to3d((P.tissue() == 2) & (np.indices((P.N, P.N))[1] > 30)),
                          end=P.to3d((P.tissue() == 2) & (np.indices((P.N, P.N))[1] < 8)))}
    th = np.linspace(0, 2 * np.pi, 121)
    circ = np.c_[np.cos(th), np.sin(th), np.zeros_like(th)]
    x0, x1, y0, y1 = 13, 26, 12, 24
    win = [(i, j) for i in range(x0, x1 + 1) for j in range(y0, y1 + 1) if P.tissue()[i, j] == 1]
    ijk = np.array([(i, j, P.Z) for i, j in win])
    tidx = chain.tindex[chain.lin(ijk)]

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 5.0), gridspec_kw=dict(wspace=0.04))
    fig.subplots_adjust(top=0.84, bottom=0.03, left=0.01, right=0.99)

    def glyphs(ax, coef, color, scale):
        B = _sh_basis(circ, coef.shape[1])
        for (i, j), c in zip(win, coef):
            r = np.maximum(c @ B, 0) * scale
            if r.max() <= 0:
                continue
            ax.fill(i + r * circ[:, 0], j + r * circ[:, 1], color=color, alpha=0.75, lw=0)

    for ax in axs:
        P.show_plane(ax)
        ax.set_xlim(x0 - 0.6, x1 + 0.6)
        ax.set_ylim(y0 - 0.6, y1 + 0.6)
    fo = sh[ijk[:, 0], ijk[:, 1], P.Z]
    B = _sh_basis(circ, fo.shape[1])
    fmax = np.maximum(fo @ B, 0).max(1, keepdims=True)
    glyphs(axs[0], fo / np.maximum(fmax, 1e-12), P.MUTED, 0.45)
    axs[0].set_title("fODF (each glyph scaled to its maximum)", loc="left", fontsize=11)
    for ax, (name, spec), col in zip(axs[1:], specs.items(), (P.ORANGE, P.BLUE)):
        res = P.segment(chain, spec)
        coef = visitation_odf(res.density_class[tidx], fo, chain.geo)
        vals = np.maximum(coef @ _sh_basis(circ, coef.shape[1]), 0)
        glyphs(ax, coef, col, 0.48 / vals.max())
        ax.set_title(f"{name}: visitation ODF (size ∝ visits)", loc="left", fontsize=11)
    fig.text(0.01, 0.93, "Where bundles cross, each visitation ODF keeps only the lobe its own walkers "
             "use", fontsize=13)
    return fig
