"""Bundle density = forward occupancy x backward completion (transition path theory's reactive
density), with both series filling in step by step."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    chain = P.build_chain()
    spec = P.arc_spec(include=False, exclude=False)            # endpoint ROIs only
    parts = P.reactive_parts(chain, spec)
    plan = parts["plan"]
    r = chain.absorb_rhs(plan.absw) * plan.mask
    pi0 = parts["pi0"]
    # column 1: this half must end in the end ROI; its weight is the partner's P(ends in start)
    c = [k for k, (T, kind) in enumerate(parts["labels"]) if kind == 1][0]
    T = 70
    S_t, B_t = [pi0[:, [c]].copy()], [r[:, [c]].copy()]
    term_f, term_b = pi0[:, [c]].copy(), r[:, [c]].copy()
    tmp = np.empty_like(term_f)
    for _ in range(T):
        chain.mul_forward(term_f, tmp)
        term_f, tmp = tmp, term_f
        S_t.append(S_t[-1] + term_f)
        chain.mul_backward(term_b, tmp)
        term_b, tmp = tmp, term_b
        B_t.append(B_t[-1] + term_b)
    S_full, B_full = parts["S"], parts["beta"]
    prod = lambda S, B: P.per_voxel(chain, (S * B)[:, 0])  # noqa: E731
    final1 = prod(S_full[:, [c]], B_full[:, [c]])
    final = P.plane(chain, parts["density"])

    cm, cmb = P.mass_cmap(), P.prob_cmap()
    s_peak = P.per_voxel(chain, S_full[:, c]).max()
    n_s = LogNorm(s_peak * 1e-3, s_peak)
    n_b = Normalize(0, 1)
    d_peak = final.max()
    n_d = LogNorm(d_peak * 1e-3, d_peak)
    rois = P.arc_rois()

    fig = plt.figure(figsize=(12.6, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=0.06, top=0.8, bottom=0.12)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    ims = []
    for ax, cmap, norm in zip(axs, (cm, cmb, cm), (n_s, n_b, n_d)):
        P.show_plane(ax)
        ims.append(ax.imshow(np.zeros((P.N, P.N)), origin="lower", cmap=cmap, norm=norm,
                             interpolation="nearest"))
        P.outline(ax, rois["start"], P.AQUA, lw=1.6)
        P.outline(ax, rois["end"], P.ORANGE if ax is not axs[1] else P.ORANGE, lw=1.6)
    P.label(axs[0], (0.5, 1.0), "start ROI", color=P.AQUA, fontsize=9)
    P.label(axs[0], (29, 1.0), "end ROI", color=P.ORANGE, fontsize=9)
    axs[0].set_title("forward: occupancy S\n(partner half ended in the start ROI)", fontsize=11)
    axs[1].set_title("backward: completion β\n(P(ends in the end ROI), best heading)", fontsize=11)
    t3 = axs[2].set_title("", fontsize=11)
    for ax, x in zip(axs[:2], (0.355, 0.645)):
        fig.text(x, 0.47, "×" if ax is axs[0] else "=", fontsize=26, ha="center", va="center",
                 color=P.INK2)
    title = fig.text(0.04, 0.93, "", fontsize=13)

    def cmask(img, norm):
        return np.ma.masked_less(img, max(norm.vmin, 1e-3 * norm.vmax)).T

    frames = []
    for k in range(T + 1):
        frames += [("t", k)] * (2 if k < 20 else 1)
    frames += [("t", T)] * 10 + [("mix", a) for a in np.linspace(0, 1, 16)] + [("mix", 1.0)] * 20

    def update(i):
        kind, v = frames[i]
        if kind == "t":
            k = v
            ims[0].set_data(cmask(P.per_voxel(chain, S_t[k][:, 0]), n_s))
            ims[1].set_data(cmask(P.per_voxel(chain, B_t[k][:, 0], "max"), n_b))
            ims[2].set_data(cmask(prod(S_t[k], B_t[k]), n_d))
            title.set_text(f"{k} steps of each pass: visits that will complete the bundle")
            t3.set_text("S·β per state, summed per voxel\n(this half's share of the bundle)")
        else:
            a = v
            ims[2].set_data(cmask((1 - a) * final1 + a * final, n_d))
            title.set_text("Add the mirror image (this half ends in the start ROI): the bundle density")
            t3.set_text("bundle density\n(both halves, plus the seed step)")

    return dict(fig=fig, update=update, frames=len(frames), fps=12, poster=len(frames) - 1)
