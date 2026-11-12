"""A small synthetic brain for the documentation's figures and animations.

Everything the docs draw is computed by MCAT itself (``mcat.Chain``, ``mcat.segment_bundles``,
``mcat.bundle_core``...) on this phantom: one slice of white matter, 2 mm voxels, sandwiched
between two slices of CSF.  Walkers never enter CSF (MCAT gives those hops zero probability and
renormalises the rest), so the whole chain lives in the plane and can be drawn in 2-D.

In the plane:

* a round brain, its mask's border (``edge``) and a two-voxel cortical ribbon (grey matter),
* white matter inside, holding
    - an **arc**: a U-shaped bundle from cortex to cortex,
    - a **stem**: a straight bundle that crosses the arc at the top,
    - radial fibres (towards the cortex) everywhere else,
* a CSF "ventricle" that walkers cannot enter.

``python phantom.py`` draws the phantom (a quick look while editing).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

N = 40                       # in-plane grid
ZOOM = 2.0                   # mm
Z = 1                        # the white-matter slice (0 and 2 are CSF)
CENTER = np.array([19.5, 19.5])
R_BRAIN = 18.5
ARC_C, ARC_R, ARC_W = np.array([19.5, 5.0]), 13.0, 2.2     # arc centre, radius, half width
STEM_X, STEM_W = 19.5, 1.6                                  # stem: vertical, x = STEM_X +- STEM_W
VENT_C, VENT_R = np.array([11.5, 25.0]), 2.6
CACHE = Path(os.environ.get("MCAT_DOCS_CACHE", Path(__file__).resolve().parents[1] / "build" / "cache"))

# ------------------------------------------------------------------ style
# Validated (dataviz skill) categorical slots, used in this fixed order; light surface.
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
SURFACE = "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"                # diverging negative pole only
ACCENT, ACCENT2, ACCENT3 = ORANGE, BLUE, AQUA
# tissue (dseg 0 outside, 1 WM, 2 GM, 3 CSF, 4 edge): quiet neutrals, so data sits on top
TISSUE_COLORS = {0: SURFACE, 1: "#ffffff", 2: "#e6e1d6", 3: "#cfe0f3", 4: "#b9b7b0"}


def style():
    """Shared matplotlib style for every docs figure."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 100, "font.size": 12, "axes.titlesize": 13,
        "axes.titleweight": "regular", "axes.titlelocation": "left", "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
        "lines.linewidth": 2.0, "axes.grid": False, "legend.frameon": False,
    })


def _ramp(name, colors):
    from matplotlib.colors import LinearSegmentedColormap

    cm = LinearSegmentedColormap.from_list(name, colors)
    cm.set_bad(alpha=0.0)
    return cm


def mass_cmap():
    """Sequential, one hue (orange), light -> dark: walker mass, occupancy, densities."""
    return _ramp("mcat_mass", ["#fdf1ea", "#f9c4a8", "#f28d5e", "#eb6834", "#b8461a", "#6e2508"])


def prob_cmap():
    """Sequential, one hue (blue), light -> dark: probabilities (completion, seed prob)."""
    return _ramp("mcat_prob", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def div_cmap():
    """Diverging blue <-> red with a neutral grey midpoint: signed terms."""
    return _ramp("mcat_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a5a4", "#e34948", "#8f1d1c"])


def tissue_rgba(alpha_wm=1.0):
    """(N, N, 4) RGBA background of the phantom's tissue."""
    from matplotlib.colors import to_rgba

    d = tissue()
    out = np.zeros(d.shape + (4,))
    for k, c in TISSUE_COLORS.items():
        out[d == k] = to_rgba(c)
    return out


def show_plane(ax, img=None, cmap=None, norm=None, tissue_bg=True, mask_below=None):
    """Draw an (N, N) plane image (x right, y up) on the tissue background."""
    ax.set_axis_off()
    ax.set_aspect("equal")
    if tissue_bg:
        ax.imshow(np.transpose(tissue_rgba(), (1, 0, 2)), origin="lower", interpolation="nearest")
    art = None
    if img is not None:
        a = np.ma.masked_invalid(np.asarray(img, float))
        if mask_below is not None:
            a = np.ma.masked_less_equal(a, mask_below)
        art = ax.imshow(a.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    return art


def outline(ax, mask, color, lw=1.6, ls="-"):
    """Outline a 2-D mask on the voxel grid."""
    m = np.pad(np.asarray(mask, bool), 1)
    segs = []
    for i in range(1, m.shape[0] - 1):
        for j in range(1, m.shape[1] - 1):
            if not m[i, j]:
                continue
            x, y = i - 1, j - 1
            if not m[i - 1, j]: segs.append([(x - .5, y - .5), (x - .5, y + .5)])   # noqa: E701
            if not m[i + 1, j]: segs.append([(x + .5, y - .5), (x + .5, y + .5)])   # noqa: E701
            if not m[i, j - 1]: segs.append([(x - .5, y - .5), (x + .5, y - .5)])   # noqa: E701
            if not m[i, j + 1]: segs.append([(x - .5, y + .5), (x + .5, y + .5)])   # noqa: E701
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, linestyles=ls, zorder=5,
                                     capstyle="round"))


def label(ax, xy, text, color=INK, **kw):
    """A direct label with a surface halo, so it reads over any data."""
    import matplotlib.patheffects as pe

    kw.setdefault("fontsize", 10)
    t = ax.annotate(text, xy, color=color, zorder=10, **kw)
    t.set_path_effects([pe.withStroke(linewidth=3, foreground=SURFACE)])
    return t


# ------------------------------------------------------------------ geometry of the phantom
def _grid():
    ij = np.indices((N, N)).astype(float)
    return ij[0], ij[1]           # x (first axis), y (second axis)


def tissue():
    """dseg in the plane: 0 outside, 1 WM, 2 GM, 3 CSF, 4 edge."""
    from scipy import ndimage

    x, y = _grid()
    brain = np.hypot(x - CENTER[0], y - CENTER[1]) <= R_BRAIN
    edge = brain & ~ndimage.binary_erosion(brain)
    inner = brain & ~edge
    gm = inner & ~ndimage.binary_erosion(inner, iterations=2)
    wm = inner & ~gm
    vent = np.hypot(x - VENT_C[0], y - VENT_C[1]) <= VENT_R
    d = np.zeros((N, N), np.uint8)
    d[wm] = 1
    d[gm] = 2
    d[wm & vent] = 3
    d[edge] = 4
    return d


def arc_mask():
    x, y = _grid()
    r = np.hypot(x - ARC_C[0], y - ARC_C[1])
    return (np.abs(r - ARC_R) <= ARC_W) & (y >= ARC_C[1] - 1)


def stem_mask():
    x, y = _grid()
    return (np.abs(x - STEM_X) <= STEM_W) & (y > ARC_C[1] + ARC_R - 4)


def fibre_field():
    """Per voxel, a list of (unit direction (3,), fraction)."""
    x, y = _grid()
    d = tissue()
    arc, stem = arc_mask(), stem_mask()
    out = {}
    for i in range(N):
        for j in range(N):
            if d[i, j] == 0:
                continue
            f = []
            if arc[i, j]:
                t = np.array([-(y[i, j] - ARC_C[1]), x[i, j] - ARC_C[0], 0.0])   # tangent
                f.append(t / np.linalg.norm(t))
            if stem[i, j]:
                f.append(np.array([0.0, 1.0, 0.0]))
            if not f:
                r = np.array([x[i, j] - CENTER[0], y[i, j] - CENTER[1], 0.0])
                n = np.linalg.norm(r)
                f.append(r / n if n > 0 else np.array([1.0, 0, 0]))
            out[i, j] = [(v, 1.0 / len(f)) for v in f]
    return out


def _sh_fit(vertices, sh_order=8):
    """(V, ncoef) least squares: SH coefficients from sphere values, in the basis
    mcat.geometry._sh_basis evaluates (DIPY descoteaux07, legacy, symmetric)."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    _, inv = sh_to_sf_matrix(Sphere(xyz=vertices), sh_order_max=sh_order, return_inv=True)
    return inv


@lru_cache(None)
def volumes():
    """(sh (N, N, 3, 45), transient (N, N, 3), absorb_class (N, N, 3), dseg (N, N)) for
    mcat.Chain.build.  Fibre fODFs are sharp Watson lobes; grey matter is isotropic."""
    from mittens.io import default_sphere

    V = np.asarray(default_sphere(), float)
    inv = _sh_fit(V)
    d = tissue()
    sh = np.zeros((N, N, 3, inv.shape[1]), np.float32)
    kappa = 30.0
    for (i, j), fibres in fibre_field().items():
        if d[i, j] == 1:
            sf = sum(w * np.exp(kappa * ((V @ v) ** 2 - 1.0)) for v, w in fibres)
        else:
            sf = np.ones(len(V))
        sh[i, j, Z] = sf @ inv
    transient = np.zeros((N, N, 3), bool)
    transient[..., Z] = d == 1
    absorb = np.zeros((N, N, 3), np.int8)
    absorb[..., Z][d == 2] = 1
    absorb[..., Z][d == 3] = 2
    absorb[..., Z][d == 4] = 3
    # CSF above and below the slice: walkers stay in the plane
    absorb[..., 0][d > 0] = 2
    absorb[..., 2][d > 0] = 2
    return sh, transient, absorb, d


def seeds():
    """White matter voxels touching grey matter (pyAFQ's WM/GM interface), (S, 3)."""
    from scipy import ndimage

    d = tissue()
    wm = d == 1
    iface = wm & ndimage.binary_dilation(d == 2, np.ones((3, 3), bool))
    ij = np.argwhere(iface)
    return np.c_[ij, np.full(len(ij), Z)]


def to3d(m2):
    m = np.zeros((N, N, 3), bool)
    m[..., Z] = m2
    return m


def arc_rois():
    """Endpoint, include and exclude ROIs for the arc (2-D masks)."""
    x, y = _grid()
    d = tissue()
    ang = np.degrees(np.arctan2(y - ARC_C[1], x - ARC_C[0]))
    cortex = d == 2
    start = cortex & (x < CENTER[0]) & (y < ARC_C[1] + 6)
    end = cortex & (x > CENTER[0]) & (y < ARC_C[1] + 6)
    near = np.abs(np.hypot(x - ARC_C[0], y - ARC_C[1]) - ARC_R) <= ARC_W + 2.5
    inc_a = (np.abs(ang - 140) < 5) & near & (d == 1)   # a plane across the left limb
    inc_b = (np.abs(ang - 40) < 5) & near & (d == 1)    # ... and the right limb
    exclude = (np.abs(y - 31) < 1.0) & (np.abs(x - STEM_X) < 4) & (d == 1)  # the stem, above the arc
    return dict(start=start, end=end, include=[inc_a, inc_b], exclude=exclude)


def arc_spec(include=True, exclude=True, ends=True, name="Arc"):
    from mcat.afq import seal_26

    r = arc_rois()
    spec = dict(name=name)
    if include:
        spec["include"] = [seal_26(to3d(m)) for m in r["include"]]
    if exclude:
        spec["exclude"] = seal_26(to3d(r["exclude"]))
    if ends:
        spec["start"], spec["end"] = to3d(r["start"]), to3d(r["end"])
    return spec


# ------------------------------------------------------------------ MCAT on the phantom
_CHAINS = {}


def build_chain(moments=False, double_odf=True, angle=35.0):
    """The real mcat.Chain on the phantom (transitions cached in ``docs/build/cache/``)."""
    from mcat import Chain, ChainConfig

    key = (moments, double_odf, angle)
    if key not in _CHAINS:
        CACHE.mkdir(parents=True, exist_ok=True)
        sh, transient, absorb, _ = volumes()
        cfg = ChainConfig(angle=angle, double_odf=double_odf, moments=moments)
        tag = f"m{int(moments)}d{int(double_odf)}a{angle:g}"
        _CHAINS[key] = Chain.build(sh, (ZOOM,) * 3, transient, absorb, cfg,
                                   cache=str(CACHE / f"phantom_{tag}.npz"))
    return _CHAINS[key]


_LAUNCHES = {}


def launches(chain):
    from mcat import prepare_launches

    if id(chain) not in _LAUNCHES:
        _LAUNCHES[id(chain)] = prepare_launches(chain, seeds())
    return _LAUNCHES[id(chain)]


def plane(chain, values, which="transient"):
    """Chain values (per transient voxel, or per absorbing voxel) -> (N, N) image."""
    return chain.to_volume(values, which)[..., Z]


def state_plane(chain, x):
    """Per-state values (n_states,) or (n_states * ncomp,) mass rows -> (N, N), summed over classes."""
    nc = chain.ncomp
    x = np.asarray(x).reshape(-1, nc)[:, 0] if nc > 1 else np.asarray(x).ravel()
    return plane(chain, x.reshape(-1, chain.nb).sum(1))


def segment(chain, spec, tol=1e-10):
    from mcat import segment_bundles

    return segment_bundles(chain, [spec], launches(chain), tol=tol, max_mem=4e9)[spec["name"]]


# ------------------------------------------------------------------ peeking inside the solves
def voxel_xy(chain, t_index):
    """(x, y) in-plane grid coordinates of transient voxel indices."""
    ijk = np.stack(np.unravel_index(chain.t_lin[np.asarray(t_index)], chain.pshape), -1) - 1
    return ijk[..., 0], ijk[..., 1]


def state_index(chain, ijk, cls):
    """State index of (voxel ijk, direction class)."""
    return int(chain.tindex[chain.lin(np.asarray(ijk)[None])[0]]) * chain.nb + int(cls)


def row_by_neighbour(chain, s):
    """One state's outgoing probabilities: (26,) to each neighbour (into white matter or
    absorbed there), and the mass that dies in white matter."""
    nb = chain.nb
    vox, cls = divmod(s, nb)
    p = np.zeros(26)
    for kk in range(chain.Q_ptr[s], chain.Q_ptr[s + 1]):
        p[chain.tmpl_n[cls, chain.Q_slot[kk]]] += chain.Q_w[kk]
    v = chain.t_lin[vox]
    for kk in range(chain.R_ptr[s], chain.R_ptr[s + 1]):
        n = int(np.flatnonzero(v + chain.noff == chain.a_lin[chain.R_ind[kk]])[0])
        p[n] += chain.R_val[kk]
    return p, float(chain.dead[s])


def launch_vector(chain, L, weights=None):
    """Walker mass right after launch: (n_states,) in white matter and (Na,) absorbed on the
    launch step.  ``weights`` (2P,) per half, default the pair weight (one streamline per seed)."""
    nb = chain.nb
    w = np.repeat(L.pair_w, 2) if weights is None else np.asarray(weights, float)
    x = np.zeros(chain.n_states)
    a = np.zeros(chain.a_lin.size)
    for h in range(L.hseed.size):
        for k in range(L.lptr[h], L.lptr[h + 1]):
            n, b = divmod(int(L.lkey[k]), nb)
            wv = int(L.hseed[h] + chain.noff[n])
            if chain.tindex[wv] >= 0:
                x[chain.tindex[wv] * nb + b] += w[h] * L.lval[k, 0]
            else:
                a[chain.aindex[wv]] += w[h] * L.lval[k, 0]
    return x, a


def reactive_parts(chain, spec, tol=1e-10, max_iter=5000):
    """The pieces mcat.solve._solve_group multiplies, for one bundle and one column at a time:
    forward occupancy S (n_states, C), completion probability beta (n_states, C), the
    columns' signs and labels, and the resulting density per voxel (Nt,).  Mirrors
    _solve_group (single-value chain), so density == segment_bundles' density_t."""
    from mcat import kernel as K
    from mcat.chain import VALID_ENDS
    from mcat.solve import _Plan, solve_series

    L = launches(chain)
    p = _Plan(chain, spec, VALID_ENDS)
    nb, Ns = chain.nb, chain.n_states
    r = chain.absorb_rhs(p.absw) * p.mask
    beta, _ = solve_series(chain.mul_backward, r.copy(), p.mask, tol, max_iter, "backward")
    P = L.pair_seed.size
    seed_ok = (~p.taboo[L.hseed][:, p.xcol]).astype(np.float64)
    vals = np.empty((L.hseed.size, p.C))
    K.half_values(L.hseed, L.lptr, L.lkey, L.lval, chain.label, chain.noff, chain.tindex,
                  chain.aindex, nb, np.ascontiguousarray(beta), p.absw, seed_ok, vals)
    V = vals.reshape(P, 2, p.nX, p.nK)
    pb = np.einsum("pxk,kl,pxl->px", V[:, 0], p.M, V[:, 1]) @ p.signs
    coef = np.einsum("phxl,kl->phxk", V[:, ::-1], p.M).reshape(2 * P, p.C)
    coef *= np.repeat(L.pair_w, 2)[:, None] * seed_ok
    pi0 = np.zeros((Ns, p.C))
    a0 = np.zeros((chain.a_lin.size, p.C))
    K.scatter_init(L.hseed, L.lptr, L.lkey, L.lval, coef, chain.label, chain.noff, chain.tindex,
                   chain.aindex, nb, p.taboo, p.xcol, pi0, a0)
    S, _ = solve_series(chain.mul_forward, pi0.copy(), p.mask, tol, max_iter, "forward")  # consumes x0
    colsign = p.signs[p.xcol]
    dens = (S * beta * colsign).reshape(chain.t_lin.size, nb, p.C).sum(1).sum(1)
    # the seed step (as _solve_group)
    pt = chain.tindex[L.seed_lin][L.pair_seed]
    on = pt >= 0
    np.add.at(dens, pt[on], (L.pair_w * pb)[on])
    labels = [(p.subsets[p.xcol[c]], int(p.kcol[c])) for c in range(p.C)]
    return dict(S=S, beta=beta, sign=colsign, labels=labels, density=dens, plan=p, pi0=pi0)


def per_voxel(chain, x, how="sum"):
    """(n_states,) -> (N, N): sum (or max) over direction classes."""
    v = np.asarray(x).reshape(-1, chain.nb)
    return plane(chain, v.sum(1) if how == "sum" else v.max(1))


def streamline_xy(chain, L, pair, halves):
    """Polyline (x, y) of a sampled streamline: half 0 reversed, the seed, half 1."""
    sx, sy = np.unravel_index(L.hseed[2 * pair], chain.pshape)[:2]
    pts = [(sx - 1, sy - 1)]
    a = [tuple(xy) for xy in zip(*voxel_xy(chain, halves[0][0] // chain.nb))]
    b = [tuple(xy) for xy in zip(*voxel_xy(chain, halves[1][0] // chain.nb))]
    return np.array(a[::-1] + pts + b, float)


# ------------------------------------------------------------------ Monte Carlo on the chain itself
def _njit():
    from numba import njit
    return njit(cache=True)


def _make_sampler():
    from numba import njit

    @njit(cache=True)
    def walk(s0, Q_ptr, Q_slot, Q_w, tn, tb, nbr_t, nb, R_ptr, R_ind, R_val, max_steps, path, u):
        """One half-walk from transient state s0, sampled from the chain's own rows.
        Fills path[:k] with the states visited (s0 first).  Returns (k, absorbing index or
        -1 for death in white matter)."""
        s = s0
        k = 0
        for it in range(max_steps):
            path[k] = s
            k += 1
            r = u[it]
            vox = s // nb
            cls = s - vox * nb
            nxt = -1
            for kk in range(Q_ptr[s], Q_ptr[s + 1]):
                r -= Q_w[kk]
                if r < 0.0:
                    sl = Q_slot[kk]
                    nxt = np.int64(nbr_t[vox, tn[cls, sl]]) * nb + tb[cls, sl]
                    break
            if nxt >= 0:
                s = nxt
                continue
            for kk in range(R_ptr[s], R_ptr[s + 1]):
                r -= R_val[kk]
                if r < 0.0:
                    return k, R_ind[kk]
            return k, -1
        return k, -1

    return walk


class MonteCarlo:
    """Classic probabilistic tracking, sampled from the same Markov chain MCAT solves exactly:
    each streamline is two half-walks from a seed, each step drawn from the chain's
    transition rows."""

    def __init__(self, chain, L, seed=0):
        self.c, self.L = chain, L
        self.rng = np.random.default_rng(seed)
        self.walk = _make_sampler()
        self.path = np.empty(20000, np.int64)
        # launch tables per half: cumulative probabilities over (key, value)
        self.lcum = [np.cumsum(L.lval[L.lptr[h]:L.lptr[h + 1], 0]) for h in range(2 * L.pair_seed.size)]
        self.pair_cum = {}
        for p, s in enumerate(L.pair_seed):
            self.pair_cum.setdefault(int(s), []).append(p)

    def _launch(self, h):
        """First exit of half h -> ('t', state) | ('a', absorbing index) | ('dead', None)."""
        c, L = self.c, self.L
        cum = self.lcum[h]
        u = self.rng.random()
        k = int(np.searchsorted(cum, u, side="right"))
        if k >= cum.size:
            return "dead", None
        key = int(L.lkey[L.lptr[h] + k])
        n, b = key // c.nb, key % c.nb
        w = int(L.hseed[h] + c.noff[n])
        if c.tindex[w] >= 0:
            return "t", int(c.tindex[w] * c.nb + b)
        return "a", int(c.aindex[w])

    def half(self, h):
        """(visited states, end absorbing index or -1)."""
        kind, v = self._launch(h)
        if kind == "dead":
            return np.zeros(0, np.int64), -1
        if kind == "a":
            return np.zeros(0, np.int64), v
        c = self.c
        u = self.rng.random(self.path.size)
        k, end = self.walk(v, c.Q_ptr, c.Q_slot, c.Q_w, c.tmpl_n, c.tmpl_b, c.nbr_t, c.nb,
                           c.R_ptr, c.R_ind, c.R_val, self.path.size, self.path, u)
        return self.path[:k].copy(), int(end)

    def streamline(self, seed_index):
        """Pick a peak pair of the seed (by its weight) and run both halves."""
        L = self.L
        pairs = self.pair_cum[seed_index]
        w = L.pair_w[pairs]
        p = pairs[int(self.rng.choice(len(pairs), p=w / w.sum()))]
        return p, [self.half(2 * p), self.half(2 * p + 1)]


def bundle_test(chain, spec):
    """f(halves) -> True if the streamline meets ``spec`` (the event solve.py computes)."""
    from mcat.chain import VALID_ENDS

    t_vox = chain.t_lin
    pad = lambda m: np.pad(m, 1).ravel() if m is not None else None  # noqa: E731
    inc = [pad(m) for m in spec.get("include", [])]
    exc = pad(spec.get("exclude"))
    start, end = pad(spec.get("start")), pad(spec.get("end"))
    a_valid = np.isin(chain.a_class, VALID_ENDS)

    def ok(halves):
        vox = [t_vox[st // chain.nb] for st, _ in halves]
        ends = [e for _, e in halves]
        if any(e < 0 or not a_valid[e] for e in ends):
            return False
        allv = np.concatenate(vox)
        if exc is not None and (exc[allv].any()):
            return False
        if any(not m[allv].any() for m in inc):
            return False
        ev = [chain.a_lin[e] for e in ends]
        if start is not None and end is not None:
            return (start[ev[0]] and end[ev[1]]) or (start[ev[1]] and end[ev[0]])
        E = start if start is not None else end
        if E is not None:
            return bool(E[ev[0]] or E[ev[1]])
        return True

    return ok


def monte_carlo_density(chain, spec, n, seed=0, checkpoints=None):
    """Sampled bundle density (expected steps per voxel, summed over seeds, as MCAT's
    density_map) from ``n`` streamlines with uniformly drawn seeds.  With ``checkpoints``,
    yields (n_so_far, density estimate) at each; else returns the final estimate."""
    L = launches(chain)
    mc = MonteCarlo(chain, L, seed)
    test = bundle_test(chain, spec)
    S = len(L.seeds)
    seed_t = chain.tindex[L.seed_lin]
    acc = np.zeros(chain.t_lin.size)
    marks = set(checkpoints or [n])
    out = []
    for i in range(1, n + 1):
        si = int(mc.rng.integers(S))
        _, halves = mc.streamline(si)
        if test(halves):
            if seed_t[si] >= 0:
                acc[seed_t[si]] += 1.0
            for st, _ in halves:
                np.add.at(acc, st // chain.nb, 1.0)
        if i in marks:
            out.append((i, acc * S / i))
    return out if checkpoints else out[-1][1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    style()
    d = tissue()
    fig, ax = plt.subplots(figsize=(5, 5))
    show_plane(ax)
    for (i, j), fib in fibre_field().items():
        if d[i, j] == 1:
            for v, w in fib:
                ax.plot([i - 0.4 * v[0], i + 0.4 * v[0]], [j - 0.4 * v[1], j + 0.4 * v[1]],
                        color=INK2, lw=0.6)
    r = arc_rois()
    for m, col in ((r["start"], BLUE), (r["end"], BLUE), (r["include"][0], AQUA),
                   (r["include"][1], AQUA), (r["exclude"], ORANGE)):
        outline(ax, m, col)
    CACHE.mkdir(exist_ok=True)
    fig.savefig(CACHE / "phantom.png", bbox_inches="tight")
    print("wrote", CACHE / "phantom.png")
