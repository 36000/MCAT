"""One state's transition probabilities, read from the real chain, as the incoming direction
changes: singleODF vs doubleODF, at the crossing of the two bundles."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from mpat.geometry import OFFSETS, _sh_basis

    vox = (22, 15, P.Z)       # next to both bundles: the neighbours disagree with this voxel
    chains = {"singleODF": P.build_chain(double_odf=False), "doubleODF": P.build_chain()}
    inplane = [int(b) for b in np.flatnonzero(OFFSETS[:, 2] == 0)]
    ang = np.degrees(np.arctan2(OFFSETS[inplane, 1], OFFSETS[inplane, 0]))
    inplane = [b for _, b in sorted(zip(ang, inplane))]          # counter-clockwise from west
    rows = {}
    for name, c in chains.items():
        rows[name] = []
        for b in inplane:
            p, dead = P.row_by_neighbour(c, P.state_index(c, vox, b))
            rows[name].append((p, dead))

    sh = P.volumes()[0][vox]
    th = np.linspace(0, 2 * np.pi, 361)
    circ = np.c_[np.cos(th), np.sin(th), np.zeros_like(th)]
    odf = np.maximum(sh @ _sh_basis(circ, sh.size), 0)
    odf /= odf.max()

    fig = plt.figure(figsize=(12, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.12, top=0.84, bottom=0.12)
    a_odf = fig.add_subplot(gs[0])
    a_odf.set_aspect("equal")
    a_odf.set_axis_off()
    a_odf.set_xlim(-1.5, 1.5)
    a_odf.set_ylim(-1.5, 1.5)
    a_odf.fill(odf * circ[:, 0], odf * circ[:, 1], color=P.MUTED, alpha=0.35, lw=0)
    a_odf.plot(odf * circ[:, 0], odf * circ[:, 1], color=P.INK2, lw=1)
    a_odf.set_title("this voxel's fODF", fontsize=11)
    arrow = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=22, color=P.ORANGE,
                            lw=3, zorder=5)
    a_odf.add_patch(arrow)
    in_lbl = P.label(a_odf, (0, -1.45), "", color=P.ORANGE, fontsize=11, ha="center")

    cm = P.mass_cmap()
    norm = Normalize(0, 0.8)
    grids = {}
    for k, name in enumerate(chains):
        ax = fig.add_subplot(gs[k + 1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.75, 1.6)
        cells, texts = {}, {}
        for n in inplane:
            dx, dy = OFFSETS[n, :2]
            r = Rectangle((dx - 0.48, dy - 0.48), 0.96, 0.96, fc=cm(0), ec=P.SURFACE, lw=2)
            ax.add_patch(r)
            cells[n] = r
            nsh = P.volumes()[0][vox[0] + dx, vox[1] + dy, P.Z]
            g = np.maximum(nsh @ _sh_basis(circ, nsh.size), 0)
            if g.max() > 0:
                g = 0.38 * g / g.max()
                ax.plot(dx + g * circ[:, 0], dy + g * circ[:, 1], color=P.INK2, lw=0.8, alpha=0.6, zorder=3)
            texts[n] = ax.text(dx, dy, "", ha="center", va="center", fontsize=11, zorder=4)
        ax.add_patch(Rectangle((-0.48, -0.48), 0.96, 0.96, fc=P.SURFACE, ec=P.INK, lw=2))
        a = FancyArrowPatch((0, 0), (0.4, 0), arrowstyle="-|>", mutation_scale=16, color=P.ORANGE, lw=2.5)
        ax.add_patch(a)
        dtx = ax.text(0, -1.68, "", ha="center", fontsize=10, color=P.INK2)
        ax.set_title(name + (": weight hops by the next voxel's fODF" if name == "doubleODF"
                             else ": first-order hops"), fontsize=11)
        grids[name] = (cells, texts, a, dtx)
    sm = plt.cm.ScalarMappable(norm, cm)
    cb = fig.colorbar(sm, ax=[fig.axes[-2], fig.axes[-1]], orientation="horizontal", fraction=0.04,
                      pad=0.02, shrink=0.5)
    cb.set_label("P(next voxel | incoming direction class)", color=P.INK2)
    cb.outline.set_visible(False)
    fig.text(0.04, 0.94, "A state is (voxel, direction class): where a walker goes next depends on "
             "where it came from", fontsize=13)

    import matplotlib.patheffects as pe

    def _halo(v):
        return [pe.withStroke(linewidth=3, foreground=cm(norm(v)))]

    names = {(-1, 0): "west", (1, 0): "east", (0, 1): "north", (0, -1): "south", (1, 1): "north-east",
             (-1, 1): "north-west", (1, -1): "south-east", (-1, -1): "south-west"}
    move, hold = 8, 14
    per = move + hold
    total = per * len(inplane)

    def update(i):
        k, f = divmod(i, per)
        t = min(f / move, 1.0)
        t = t * t * (3 - 2 * t)
        j = (k + 1) % len(inplane)
        b0, b1 = inplane[k], inplane[j]
        d = (1 - t) * OFFSETS[b0, :2] / np.linalg.norm(OFFSETS[b0, :2]) + \
            t * OFFSETS[b1, :2] / np.linalg.norm(OFFSETS[b1, :2])
        d /= np.linalg.norm(d)
        arrow.set_positions((-1.25 * d[0], -1.25 * d[1]), (1.25 * d[0], 1.25 * d[1]))
        cur = b1 if t > 0.5 else b0
        in_lbl.set_text(f"incoming: heading {names[tuple(OFFSETS[cur, :2])]}")
        for name in chains:
            cells, texts, a, dtx = grids[name]
            p0, d0 = rows[name][k]
            p1, d1 = rows[name][j]
            p = (1 - t) * p0 + t * p1
            dead = (1 - t) * d0 + t * d1
            for n in inplane:
                cells[n].set_facecolor(cm(norm(p[n])))
                texts[n].set_text(f"{p[n]:.2f}" if p[n] >= 0.005 else "")
                texts[n].set_color(P.SURFACE if p[n] > 0.45 else P.INK)
                texts[n].set_path_effects(_halo(p[n]))
            a.set_positions((-0.3 * d[0], -0.3 * d[1]), (0.3 * d[0], 0.3 * d[1]))
            dtx.set_text(f"dies in white matter: {dead:.2f}" if dead >= 0.005 else "")

    return dict(fig=fig, update=update, frames=total, fps=12, poster=per * 2 + move)
