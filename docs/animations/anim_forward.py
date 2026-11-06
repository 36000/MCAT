"""Forward pass: walker mass from every seed's launch, one step of the chain at a time."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    chain = P.build_chain()
    L = P.launches(chain)
    x, a0 = P.launch_vector(chain, L)
    start = x.sum() + a0.sum()
    absorbed = a0.copy()
    total = x.copy()
    dead = 0.0
    tol = 1e-10
    steps = [(0, x.copy(), total.copy(), absorbed.copy(), dead, x.sum() / start)]
    term = x[:, None].copy()
    nxt = np.empty_like(term)
    t = 0
    while term.sum() / start > tol and t < 400:
        absorbed += chain.absorbed_from(term)[:, 0]
        dead += float((chain.dead * term[:, 0]).sum())
        chain.mul_forward(term, nxt)
        term, nxt = nxt, term
        t += 1
        total += term[:, 0]
        steps.append((t, term[:, 0].copy(), total.copy(), absorbed.copy(), dead, term.sum() / start))

    cm, cmb = P.mass_cmap(), P.prob_cmap()
    occ_peak = P.state_plane(chain, total).max()
    n_tr = LogNorm(occ_peak * 1e-4, occ_peak * 0.3)
    n_occ = LogNorm(occ_peak * 1e-3, occ_peak)
    ab_plane = lambda a: P.plane(chain, a, "absorbing")  # noqa: E731
    ab_peak = ab_plane(steps[-1][3]).max()
    n_ab = LogNorm(ab_peak * 1e-3, ab_peak)

    fig = plt.figure(figsize=(12.4, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.1, top=0.84, bottom=0.14)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    P.show_plane(ax1, None)
    im1 = ax1.imshow(np.zeros((P.N, P.N)), origin="lower", cmap=cm, norm=n_tr, interpolation="nearest")
    ax1.set_title("in transit at step t")
    P.show_plane(ax2, None)
    im2 = ax2.imshow(np.zeros((P.N, P.N)), origin="lower", cmap=cm, norm=n_occ, interpolation="nearest")
    im2b = ax2.imshow(np.zeros((P.N, P.N)), origin="lower", cmap=cmb, norm=n_ab, interpolation="nearest")
    ax2.set_title("occupancy so far")
    P.label(ax2, (-0.5, -2.5), "blue: walkers absorbed in grey matter / edge", color=P.BLUE, fontsize=9,
            annotation_clip=False)
    P.label(ax1, (8.5, 27.5), "CSF: never\nentered", color=P.INK2, fontsize=9, ha="center")
    title = fig.text(0.04, 0.94, "", fontsize=13)

    ts = np.array([s[0] for s in steps])
    rem = np.array([s[5] for s in steps])
    ax3.set_yscale("log")
    ax3.set_xlim(0, ts[-1] + 2)
    ax3.set_ylim(tol / 10, 2)
    ax3.axhline(tol, color=P.MUTED, lw=1.2, ls="--")
    P.label(ax3, (ts[-1] * 0.03, tol * 2.2), "tol = 1e-10: stop", color=P.INK2, fontsize=10)
    (curve,) = ax3.plot([], [], color=P.ORANGE)
    (dot,) = ax3.plot([], [], "o", color=P.ORANGE, ms=8, mec=P.SURFACE, mew=2)
    ax3.set_xlabel("step t")
    ax3.set_title("walker mass still moving")
    ax3.yaxis.tick_right()
    ax3.spines["right"].set_visible(True)
    ax3.spines["left"].set_visible(False)
    share = ax3.text(0.04, 0.5, "", transform=ax3.transAxes, ha="left", va="center", fontsize=10,
                     color=P.INK2)

    def mask(img, norm):
        return np.ma.masked_less(img, norm.vmin).T

    frames = []
    for k, s in enumerate(steps):
        frames += [k] * (3 if s[0] < 12 else 1)
    frames += [len(steps) - 1] * 18

    def update(i):
        k = frames[i]
        t, term_, tot, ab, dd, r = steps[k]
        im1.set_data(mask(P.state_plane(chain, term_), n_tr))
        im2.set_data(mask(P.state_plane(chain, tot), n_occ))
        im2b.set_data(mask(ab_plane(ab), n_ab))
        title.set_text(f"Forward pass, step {t}: every seed's walkers move at once, "
                       "as expected mass (no sampling)")
        curve.set_data(ts[:k + 1], rem[:k + 1])
        dot.set_data([t], [max(r, tol / 10)])
        share.set_text(f"absorbed in grey matter / edge: {ab.sum() / start:6.1%}\n"
                       f"died in white matter: {dd / start:6.1%}\nstill moving: {r:.1e}")

    return dict(fig=fig, update=update, frames=len(frames), fps=12, poster=36 + 12)
