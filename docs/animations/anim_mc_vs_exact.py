"""Monte Carlo tracking converges, noisily, to the density MPAT computes exactly."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    chain = P.build_chain()
    L = P.launches(chain)
    spec = P.arc_spec(include=False, exclude=False)            # endpoint ROIs only
    exact = P.segment(chain, spec).density_t
    mc = P.MonteCarlo(chain, L, seed=1)
    test = P.bundle_test(chain, spec)
    S = len(L.seeds)
    seed_t = chain.tindex[L.seed_lin]

    marks = np.unique(np.round(np.logspace(1, 5, 64)).astype(int))
    acc = np.zeros(chain.t_lin.size)
    frames, lines, n_done = [], [], 0
    for m in marks:
        while n_done < m:
            si = int(mc.rng.integers(S))
            pair, halves = mc.streamline(si)
            n_done += 1
            if test(halves):
                acc[seed_t[si]] += 1.0
                for st, _ in halves:
                    np.add.at(acc, st // chain.nb, 1.0)
                lines.append(P.streamline_xy(chain, L, pair, halves))
                lines = lines[-6:]
        est = acc * S / m
        err = np.abs(est - exact).sum() / exact.sum()
        frames.append((m, est.copy(), err, list(lines)))

    peak = exact.max()
    norm = LogNorm(peak * 1e-3, peak)
    cm = P.mass_cmap()
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.08)
    a_mc, a_ex, a_err = (fig.add_subplot(gs[i]) for i in range(3))
    floor = peak * 1e-3
    im_mc = P.show_plane(a_mc, P.plane(chain, frames[0][1]), cm, norm, mask_below=floor)
    P.show_plane(a_ex, P.plane(chain, exact), cm, norm, mask_below=floor)
    rois = P.arc_rois()
    for a in (a_mc, a_ex):
        P.outline(a, rois["start"] | rois["end"], P.BLUE, lw=1.2)
    t_mc = a_mc.set_title("")
    a_ex.set_title("MPAT: exact")
    P.label(a_ex, (30, 2.5), "endpoint\nROIs", color=P.BLUE, fontsize=9)
    trails = [a_mc.plot([], [], color=P.INK, lw=1.0, alpha=0.75, zorder=6)[0] for _ in range(6)]

    ns = np.array([f[0] for f in frames])
    errs = np.array([f[2] for f in frames])
    a_err.set_xscale("log")
    a_err.set_yscale("log")
    a_err.set_xlim(8, 1.4e5)
    a_err.set_ylim(errs.min() * 0.5, max(errs.max() * 1.5, 2))
    ref = errs[len(errs) // 2] * np.sqrt(ns[len(errs) // 2] / ns)
    a_err.plot(ns, ref, color=P.MUTED, lw=1.2, ls="--")
    P.label(a_err, (ns[8], ref[8] * 1.25), "∝ 1/√N", color=P.INK2, fontsize=10)
    (curve,) = a_err.plot([], [], color=P.ORANGE, lw=2)
    (dot,) = a_err.plot([], [], "o", color=P.ORANGE, ms=8, mec=P.SURFACE, mew=2)
    a_err.set_xlabel("streamlines sampled, N")
    a_err.set_title("Monte Carlo error")
    a_err.text(0.03, 0.03, "relative L1 error\nMPAT: 0 (no sampling)", transform=a_err.transAxes,
               color=P.INK2, fontsize=10, va="bottom")
    a_err.yaxis.set_label_position("right")
    a_err.yaxis.tick_right()
    a_err.spines["right"].set_visible(True)
    a_err.spines["left"].set_visible(False)
    cb = fig.colorbar(im_mc, ax=[a_mc, a_ex], orientation="horizontal", fraction=0.05, pad=0.03,
                      shrink=0.6)
    cb.set_label("expected visits per voxel (log)", color=P.INK2)
    cb.outline.set_visible(False)

    hold = 18

    def update(i):
        k = min(i, len(frames) - 1)
        m, est, err, ls = frames[k]
        im_mc.set_data(np.ma.masked_less_equal(P.plane(chain, est), floor).T)
        t_mc.set_text(f"Monte Carlo, N = {m:,}")
        for tr, xy in zip(trails, ls + [None] * 6):
            tr.set_data(*(xy.T if xy is not None else ([], [])))
        curve.set_data(ns[:k + 1], errs[:k + 1])
        dot.set_data([m], [err])

    return dict(fig=fig, update=update, frames=len(frames) + hold, fps=10, poster=len(frames) - 1)
