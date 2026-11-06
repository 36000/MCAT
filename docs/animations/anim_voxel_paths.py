"""Hops inside one voxel: MITTENS's exact start-point volumes, from mcat.geometry.build_geometry."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from mcat.geometry import OFFSETS, build_geometry

    g = build_geometry((P.ZOOM,) * 3, 35.0)
    V = g.vertices
    inplane = np.abs(V[:, 2]) < 1e-9
    d1 = int(np.flatnonzero(inplane & (np.abs(np.degrees(np.arctan2(V[:, 1], V[:, 0])) - 36) < 1))[0])
    u = V[d1, :2]
    mine = g.p_dirs[:, 0] == d1
    one = np.flatnonzero(mine & (g.p_len == 1))
    two_same = np.flatnonzero(mine & (g.p_len == 2) & (g.p_dirs[:, 1] == d1))
    cone = g.cone_idx[g.cone_ptr[d1]:g.cone_ptr[d1 + 1]]
    two_all = np.flatnonzero(mine & (g.p_len == 2))

    nb_name = {(1, 0, 0): "east", (1, 1, 0): "north-east", (0, 1, 0): "north"}
    nb_col = {(1, 0, 0): P.BLUE, (1, 1, 0): P.ORANGE, (0, 1, 0): P.AQUA}

    # start points on a grid (mid-depth), and their fates after one hop
    k = 24
    gx, gy = np.meshgrid((np.arange(k) + 0.5) / k, (np.arange(k) + 0.5) / k)
    pts = np.c_[gx.ravel(), gy.ravel()]
    land1 = pts + u
    fate1 = np.floor(land1).astype(int)
    stays = np.all(fate1 == 0, 1)
    land2 = land1 + u
    fate2 = np.floor(land2).astype(int)
    fate = np.where(stays[:, None], fate2, fate1)
    col = np.array([nb_col.get((a, b, 0), P.MUTED) for a, b in fate])

    # P(exit neighbour | first hop d1) under a uniform fODF: one-hop volumes + two-hop / |cone|
    probs = {}
    for p in one:
        key = tuple(OFFSETS[g.p_nbr[p]])
        probs[key] = probs.get(key, 0) + g.p_vol[p]
    for p in two_all:
        key = tuple(OFFSETS[g.p_nbr[p]])
        probs[key] = probs.get(key, 0) + g.p_vol[p] / cone.size
    order = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]
    other = sum(v for kk, v in probs.items() if kk not in order)

    fig = plt.figure(figsize=(11, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.25, top=0.86, bottom=0.2)
    ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 2.25)
    ax.set_ylim(-0.35, 2.25)
    ax.set_axis_off()
    for i in range(-1, 3):
        ax.plot([i, i], [-0.35, 2.25], color=P.MUTED, lw=0.6, alpha=0.5)
        ax.plot([-0.35, 2.25], [i, i], color=P.MUTED, lw=0.6, alpha=0.5)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=P.INK, lw=2, zorder=4))
    P.label(ax, (0.02, -0.17), "voxel v", color=P.INK, fontsize=11)
    for key, (x, y) in (((1, 0, 0), (1.5, 0.5)), ((1, 1, 0), (1.5, 1.5)), ((0, 1, 0), (0.5, 1.5))):
        P.label(ax, (x, y), nb_name[key], color=nb_col[key], fontsize=11, ha="center", va="center")
    arrow = FancyArrowPatch((0.5 - 0.5 * u[0], 0.5 - 0.5 * u[1]), (0.5 + 0.5 * u[0], 0.5 + 0.5 * u[1]),
                            arrowstyle="-|>", mutation_scale=22, color=P.INK, lw=2.5, zorder=8)
    ax.add_patch(arrow)
    cone_lines = [ax.plot([], [], color=P.INK2, lw=0.8, alpha=0.0, zorder=7)[0] for _ in cone]
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=9, c=P.MUTED, zorder=6, lw=0)
    title = fig.text(0.04, 0.95, "", fontsize=13, color=P.INK)

    boxes = []
    for p in list(one) + list(two_same):
        key = tuple(OFFSETS[g.p_nbr[p]])
        lo, hi = g.lo[p], g.hi[p]
        r = Rectangle(lo[:2], *(hi - lo)[:2], fc=nb_col[key], ec=P.SURFACE, lw=2, alpha=0.0, zorder=3)
        ax.add_patch(r)
        tx = P.label(ax, ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2),
                     f"{'2 hops' if g.p_len[p] == 2 else '1 hop'}\nvol {g.p_vol[p]:.3f}",
                     color=P.INK, fontsize=9, ha="center", va="center")
        tx.set_alpha(0.0)
        boxes.append((r, tx, g.p_len[p]))

    axb.set_title("P(exit neighbour | first hop)")
    names = ["east", "north-east", "north", "other\n(out of plane)"]
    vals = [probs.get(kk, 0) for kk in order] + [other]
    cols = [nb_col[kk] for kk in order] + [P.MUTED]
    bars = axb.barh(range(4), [0] * 4, color=cols, height=0.6)
    axb.set_yticks(range(4), names)
    axb.invert_yaxis()
    axb.set_xlim(0, 0.65)
    axb.set_xlabel("probability (uniform fODF)")
    axb.tick_params(axis="y", length=0)
    vtxt = [axb.text(0, i, "", va="center", fontsize=10, color=P.INK2) for i in range(4)]
    axb.text(0.0, -0.2, f"This first hop: {mine.sum()} MITTENS paths; second hops turn within\n"
             f"a {35:g}° cone of {cone.size} sphere directions. Grid: one voxel, seen from above.",
             transform=axb.transAxes, fontsize=9, color=P.INK2, va="top")

    def ease(t):
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)

    F = dict(intro=12, hop1=20, color=12, back=12, box1=16, hop2=22, box2=14, bars=14, hold=24)
    starts = np.cumsum([0] + list(F.values()))
    ph = dict(zip(F, starts[:-1]))
    total = int(starts[-1])

    def update(i):
        def frac(name):
            return ease((i - ph[name]) / F[name])

        if i < ph["hop1"]:
            title.set_text("A walker enters moving this way; every start point in the voxel...")
            xy = pts
            c = [P.MUTED] * len(pts)
        elif i < ph["back"]:
            t = frac("hop1")
            xy = pts + t * u
            c = col if i >= ph["color"] else [P.MUTED] * len(pts)
            c = np.where(stays, P.MUTED, col) if i >= ph["color"] else c
            title.set_text("...takes a hop one voxel long; where it lands decides the neighbour")
        elif i < ph["hop2"]:
            t = frac("back")
            xy = np.where(stays[:, None], pts + u * (1 - 0.0), pts + u * (1 - t))
            xy = np.where(stays[:, None], pts + u, xy)
            c = np.where(stays, P.MUTED, col)
            title.set_text("Start points sharing a fate fill a box: its volume is exact")
        else:
            t = frac("hop2")
            xy = np.where(stays[:, None], pts + u * (1 + t), pts)
            c = np.where(stays & (t < 1), P.MUTED, col)
            title.set_text("Hops that stay inside turn (within the cone) and hop again")
            if i >= ph["box2"]:
                xy = pts
                title.set_text("Every path: a hop sequence, its exit, and the volume that takes it")
        arrow.set_alpha(1.0 - (frac("back") if i >= ph["back"] else 0.0))
        sc.set_offsets(xy)
        sc.set_color(c)
        a1 = frac("box1") if i >= ph["box1"] else 0.0
        a2 = frac("box2") if i >= ph["box2"] else 0.0
        for r, tx, L in boxes:
            a = a1 if L == 1 else a2
            r.set_alpha(0.28 * a)
            tx.set_alpha(a)
        # the cone of second-hop directions, from the stayers' centre
        ca = 0.5 * (frac("hop2") if ph["hop2"] <= i < ph["box2"] else 0.0)
        c0 = np.array([0.095, 0.206]) + u
        for ln, d in zip(cone_lines, cone):
            ln.set_data([c0[0], c0[0] + 0.35 * V[d, 0]], [c0[1], c0[1] + 0.35 * V[d, 1]])
            ln.set_alpha(ca)
        b = frac("bars") if i >= ph["bars"] else 0.0
        for bar, v, tx in zip(bars, vals, vtxt):
            bar.set_width(v * b)
            tx.set_x(v * b + 0.01)
            tx.set_text(f"{v:.3f}" if b > 0.99 else "")

    return dict(fig=fig, update=update, frames=total, fps=12, poster=total - 1)
