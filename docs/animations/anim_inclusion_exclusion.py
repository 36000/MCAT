"""Include ROIs by inclusion-exclusion: P(hit A and B) = P(any) - P(avoid A) - P(avoid B)
+ P(avoid A and B), each term an exclusion problem (one column of the solve)."""
import numpy as np

import phantom as P


def make():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm

    chain = P.build_chain()
    spec = P.arc_spec(exclude=False, ends=False, name="AB")      # two include ROIs only
    parts = P.reactive_parts(chain, spec)
    nb = chain.nb
    terms = (parts["S"] * parts["beta"]).reshape(chain.t_lin.size, nb, -1).sum(1)   # (Nt, C)
    signs = parts["sign"]
    subsets = [T for T, _ in parts["labels"]]
    names = {(): "all walkers", (0,): "avoiding A", (1,): "avoiding B", (0, 1): "avoiding A and B"}
    planes = [P.plane(chain, terms[:, k]) for k in range(len(subsets))]
    partial = np.cumsum([s * p for s, p in zip(signs, planes)], axis=0)
    final = P.plane(chain, parts["density"])

    rois = P.arc_rois()
    inc = rois["include"]
    peak = max(p.max() for p in planes)
    n_term = LogNorm(peak * 1e-3, peak)
    M = np.abs(partial).max()
    n_sum = SymLogNorm(linthresh=final.max() * 1e-2, vmin=-M, vmax=M, base=10)
    cm, cd = P.mass_cmap(), P.div_cmap()

    fig = plt.figure(figsize=(12.4, 6.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.15, 1.15], wspace=0.05, hspace=0.25,
                          top=0.86, bottom=0.04, left=0.02, right=0.95)
    small = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    big = fig.add_subplot(gs[:, 2:])
    t_ims = []
    for ax, k in zip(small, range(4)):
        P.show_plane(ax)
        t_ims.append(ax.imshow(np.ma.masked_less(planes[k], n_term.vmin).T, origin="lower", cmap=cm,
                               norm=n_term, interpolation="nearest", alpha=0.25))
        for m in inc:
            P.outline(ax, m, P.AQUA, lw=1.2)
        sign = "+" if signs[k] > 0 else "−"
        ax.set_title(f"{sign}  {names[subsets[k]]}", fontsize=12, color=P.INK)
    P.label(small[0], (5.5, 16.5), "A", color=P.AQUA, fontsize=11, fontweight="bold")
    P.label(small[0], (31.5, 16.5), "B", color=P.AQUA, fontsize=11, fontweight="bold")
    P.show_plane(big)
    big_im = big.imshow(np.zeros((P.N, P.N)), origin="lower", cmap=cd, norm=n_sum, interpolation="nearest")
    for m in inc:
        P.outline(big, m, P.AQUA, lw=1.6)
    btitle = big.set_title("", fontsize=12)
    cb = fig.colorbar(big_im, ax=big, fraction=0.035, pad=0.01)
    cb.outline.set_visible(False)
    title = fig.text(0.02, 0.94, "", fontsize=13)

    hold_in, ramp = 10, 10
    seq = []
    for k in range(4):
        seq += [("term", k, a) for a in np.linspace(0, 1, ramp)] + [("term", k, 1.0)] * hold_in
    seq += [("final", 3, a) for a in np.linspace(0, 1, 12)] + [("final", 3, 1.0)] * 24

    def update(i):
        kind, k, a = seq[i]
        for j, im in enumerate(t_ims):
            im.set_alpha(1.0 if j <= k else 0.25)
            small[j].title.set_fontweight("bold" if j == k and kind == "term" else "normal")
        if kind == "term":
            prev = partial[k - 1] if k else np.zeros_like(final)
            img = prev + a * signs[k] * planes[k]
            title.set_text("Include ROIs: P(hits A and B) = P(all) − P(avoid A) − P(avoid B) "
                           "+ P(avoid both)")
            btitle.set_text(f"running sum after {k + 1} of 4 terms")
            big_im.set_cmap(cd)
            big_im.set_norm(n_sum)
            cb.set_ticks([])
            cb.set_label("running sum (symmetric log): red +, grey 0, blue −", color=P.INK2)
        else:
            img = final
            title.set_text("Each term is an exclusion problem; their signed sum is the bundle "
                           "through both A and B")
            btitle.set_text("bundle density: through A and B")
            big_im.set_cmap(cm)
            big_im.set_norm(LogNorm(final.max() * 1e-3, final.max()))
            cb.set_ticks([])
            cb.set_label("expected visits per voxel (log, 1e-3 of peak to peak)", color=P.INK2)
            cb.ax.minorticks_off()
            img = np.where(final >= final.max() * 1e-3 * a + 1e-300, final, np.nan)
        big_im.set_data(np.ma.masked_invalid(np.where(np.abs(img) > 0, img, np.nan)).T)

    return dict(fig=fig, update=update, frames=len(seq), fps=12, poster=len(seq) - 1)
