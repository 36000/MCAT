"""The absorbing Markov chain on (white-matter voxel, direction class) states.

Transitions are stored as a stencil: a state's transition goes to one of its voxel's 26
neighbours, arriving in some class, and which (neighbour, class) pairs a class can reach is
fixed by the geometry (about 150 of 676).  Each transition is a 1-byte slot id into its
class's template plus a float64 weight; one copy serves forward (scatter) and backward
(gather) steps.  Absorption (into grey matter, CSF, the brain edge or outside) is a
separate sparse block.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import kernel as K
from .geometry import OFFSETS, build_geometry, pmf_and_cone_mass, seed_peaks

log = logging.getLogger("mcat")

ABSORB_CLASSES = ("outside", "gm", "csf", "edge")  # tissue class ids 0..3 (csf: never entered)
CSF = ABSORB_CLASSES.index("csf")
VALID_ENDS = (1, 3)                                 # grey matter and brain edge


@dataclass
class ChainConfig:
    angle: float = 35.0               # maximum turn per hop, degrees
    step: float = 1.0                 # hop length, voxels (of the smallest voxel side)
    double_odf: bool = True           # weight each hop by the fODF where it lands (iFOD2-like)
    moments: bool = False             # spatial first moments (sub-voxel position, see moments.py)
    moment_closure: str = "standard"  # "standard", "positivity" (alpha=1/sqrt(3)), or "transverse" (2D plane)
    stop_mass: float = 0.0            # MITTENS stopping regularizer tau in Z_v(e) + tau
    chunk: int = 256                  # voxels per build chunk

    def key(self):
        d = asdict(self)
        d.pop("chunk")
        if d.get("stop_mass", 0.0) == 0.0:
            d.pop("stop_mass", None)
        if d.get("moment_closure", "standard") == "standard":
            d.pop("moment_closure", None)
        return d


def class_templates(geo):
    """Per direction class, the (neighbour, arrival class) pairs it can reach: (26, S)
    neighbour and class arrays (-1 padded) and the (26, 676) slot map."""
    nb = geo.nb
    keys = geo.t_nbr * nb + geo.t_bin
    sets = []
    for b in range(nb):
        es = geo.bin_idx[geo.bin_ptr[b]:geo.bin_ptr[b + 1]]
        ds = np.unique(np.concatenate([geo.cone_idx[geo.cone_ptr[e]:geo.cone_ptr[e + 1]] for e in es]))
        sets.append(np.unique(np.concatenate([keys[geo.d1_ptr[d]:geo.d1_ptr[d + 1]] for d in ds])))
    S = max(s.size for s in sets)
    tn = np.full((nb, S), -1, np.int16)
    tb = np.full((nb, S), -1, np.int16)
    slot_of = np.full((nb, 26 * nb), -1, np.int16)
    for b, s in enumerate(sets):
        tn[b, :s.size], tb[b, :s.size] = s // nb, s % nb
        slot_of[b, s] = np.arange(s.size)
    return tn, tb, slot_of


class _Grower:
    """Append-only typed storage with extrapolated capacity (no second full copy)."""

    def __init__(self, dtype, width=None):
        self.n = 0
        self.a = np.empty((0,) if width is None else (0, width), dtype)

    def add(self, x, frac_done):
        need = self.n + len(x)
        if need > len(self.a):
            est = int(need / max(frac_done, 1e-3) * 1.3) if frac_done < 1 else need
            b = np.empty((max(need, est, int(len(self.a) * 1.3)),) + self.a.shape[1:], self.a.dtype)
            b[:self.n] = self.a[:self.n]
            self.a = b
        self.a[self.n:need] = x
        self.n = need

    def array(self):
        return self.a[:self.n].copy() if len(self.a) - self.n > self.n // 50 else self.a[:self.n]


class Chain:
    ncomp = 1  # values carried per state (the moment chain carries 4)

    @classmethod
    def build(cls, sh, zooms, transient, absorb_class, cfg: ChainConfig = ChainConfig(), cache=None):
        """sh            (X, Y, Z, C) SH coefficients (DIPY descoteaux07; symmetric or full basis)
        zooms         voxel sizes (mm)
        transient     (X, Y, Z) bool: white matter, where walkers move
        absorb_class  (X, Y, Z) int: 0 outside, 1 GM, 2 CSF, 3 brain edge (where not transient).
                      GM and edge absorb walkers; CSF has zero transition probability (walkers
                      never step there; their other hops are renormalised)
        cache         optional .npz: transitions are loaded from it when inputs and config
                      match, else built and saved there"""
        if cfg.moments and cls is Chain:
            from .moments import MomentChain
            return MomentChain.build(sh, zooms, transient, absorb_class, cfg, cache)
        self = cls()
        t0 = time.time()
        self.cfg = cfg
        self.shape = tuple(transient.shape)
        self.pshape = pshape = tuple(s + 2 for s in self.shape)
        self.noff = (OFFSETS @ np.array([pshape[1] * pshape[2], pshape[2], 1])).astype(np.int64)
        trans_p = np.zeros(pshape, bool)
        trans_p[1:-1, 1:-1, 1:-1] = transient
        cls_p = np.zeros(pshape, np.int8)
        cls_p[1:-1, 1:-1, 1:-1] = absorb_class
        label = np.zeros(pshape, np.int8)
        label[cls_p > 0] = K.ABSORBING
        label[cls_p == CSF] = K.BLOCKED
        label[trans_p] = K.TRANSIENT
        self.label = label.ravel()
        self.t_lin = np.flatnonzero(trans_p.ravel()).astype(np.int64)
        self.tindex = np.full(label.size, -1, np.int64)
        self.tindex[self.t_lin] = np.arange(self.t_lin.size)
        a_mask = ndimage.binary_dilation(trans_p, np.ones((3, 3, 3), bool)) & ~trans_p
        self.a_lin = np.flatnonzero(a_mask.ravel()).astype(np.int64)
        self.aindex = np.full(label.size, -1, np.int64)
        self.aindex[self.a_lin] = np.arange(self.a_lin.size)
        self.a_class = cls_p.ravel()[self.a_lin].astype(np.int64)

        self.geo = geo = build_geometry(zooms, cfg.angle, cfg.step)
        self.nb = geo.nb
        need = np.flatnonzero((trans_p | (a_mask & (label.reshape(pshape) == K.ABSORBING))).ravel())
        sh_p = np.zeros(pshape + (sh.shape[-1],), np.float32)
        sh_p[1:-1, 1:-1, 1:-1] = sh
        rows = sh_p.reshape(-1, sh.shape[-1])[need]
        del sh_p
        has = np.any(rows != 0, axis=1)
        need, rows = need[has], rows[has]
        self.X, self.Y = pmf_and_cone_mass(rows, geo)
        del rows
        self.xrow = np.full(label.size, -1, np.int64)
        self.xrow[need] = np.arange(need.size)
        log.info("fODFs for %d voxels (%d white matter), %d paths, in %.1fs", need.size,
                 self.t_lin.size, geo.p_vol.size, time.time() - t0)

        fp = self._fingerprint(sh, zooms, transient, absorb_class)
        if not (cache is not None and self._load_cache(cache, fp)):
            self._build_matrices()
            if cache is not None:
                self._save_cache(cache, fp)
        self._finalize()
        log.info("chain: %d states, %d transitions, %.1f GB, built in %.0fs", self.n_states,
                 self.n_transitions, self.nbytes / 1e9, time.time() - t0)
        return self

    # ------------------------------------------------------------------ build
    def _kargs(self):
        g = self.geo
        return (self.X, self.Y, self.xrow, self.label, self.noff, self.cfg.double_odf, g.p_vol,
                g.p_dirs, g.p_len, g.p_nbr, g.p_trip, g.d1_ptr, g.t_nbr, g.t_bin, g.cone_ptr, g.cone_idx)

    def _kargs_tree(self):
        g = self.geo
        return (self.X, self.Y, self.xrow, self.label, self.noff, self.cfg.double_odf, g.u_dir,
                g.u_parent, g.u_d1, g.u_skip, g.u_eptr, g.e_vol, g.e_nbr, g.e_trip, g.d1_ptr,
                g.t_nbr, g.t_bin, g.cone_ptr, g.cone_idx)

    _CACHE_KEYS = ("Q_ptr", "Q_slot", "Q_w", "R_ptr", "R_ind", "R_val", "dead", "tmpl_n", "tmpl_b")

    def _build_matrices(self):
        nb, Nt = self.nb, self.t_lin.size
        tn, tb, slot_of = class_templates(self.geo)
        self.tmpl_n, self.tmpl_b = tn, tb
        targs = self._kargs_tree()
        tau = float(self.cfg.stop_mass)
        Ns = Nt * nb
        q_slot, q_w = _Grower(np.uint8 if tn.shape[1] <= 256 else np.uint16), _Grower(np.float64)
        r_ind, r_w = _Grower(np.int32), _Grower(np.float64)
        Q_ptr = np.zeros(Ns + 1, np.int64)
        R_ptr = np.zeros(Ns + 1, np.int64)
        self.dead = np.empty(Ns)
        chunk, tt = self.cfg.chunk, time.time()
        for s in range(0, Nt, chunk):
            vl = self.t_lin[s:s + chunk]
            out = np.empty((vl.size, nb, 26, nb))
            dead = np.empty((vl.size, nb))
            K.chunk_blocks_tree(vl, *targs, self.geo.bin_ptr, self.geo.bin_idx, out, dead, tau)
            qp, qs, qw, rp, ri, rw, dd = K.stencil_rows(vl, out, dead, self.label, self.noff, self.aindex, slot_of)
            del out
            r0, r1 = s * nb, (s + vl.size) * nb
            Q_ptr[r0 + 1:r1 + 1] = qp[1:] + q_w.n
            R_ptr[r0 + 1:r1 + 1] = rp[1:] + r_w.n
            frac = (s + vl.size) / Nt
            q_slot.add(qs, frac), q_w.add(qw, frac), r_ind.add(ri, frac), r_w.add(rw, frac)
            self.dead[r0:r1] = dd
            if s and (s // chunk) % 40 == 0:
                el = time.time() - tt
                log.info("  transitions %d/%d voxels (%.0fs, eta %.0fs)", s, Nt, el, el / s * (Nt - s))
        self.Q_ptr, self.R_ptr = Q_ptr, R_ptr
        self.Q_slot, self.Q_w = q_slot.array(), q_w.array()
        self.R_ind, self.R_val = r_ind.array(), r_w.array()

    def _finalize(self):
        self.n_states = self.t_lin.size * self.nb
        self.nbr_t = self.tindex[self.t_lin[:, None] + self.noff[None, :]].astype(np.int32)
        self.chunk_ptr = K.slab_chunks(self.t_lin, self.pshape, self.nb)
        self.n_transitions = int(self.Q_w.size)
        self.nbytes = sum(getattr(self, k).nbytes for k in self._CACHE_KEYS + ("nbr_t",))

    def _fingerprint(self, sh, zooms, transient, absorb_class):
        h = hashlib.sha1(json.dumps(self.cfg.key(), sort_keys=True).encode())
        h.update(type(self).__name__.encode())
        h.update(b"csf-blocked")   # transitions changed: caches built before ignored CSF blocking
        for a in (np.ascontiguousarray(sh, np.float32), np.asarray(zooms, np.float64),
                  transient.astype(np.uint8), np.asarray(absorb_class, np.int8)):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def _save_cache(self, path, fp):
        np.savez(path, fingerprint=np.array(fp), **{k: getattr(self, k) for k in self._CACHE_KEYS})

    def _load_cache(self, path, fp):
        if not Path(path).exists():
            return False
        z = np.load(path)
        if str(z["fingerprint"]) != fp or any(k not in z.files for k in self._CACHE_KEYS):
            log.info("cache %s does not match the inputs; rebuilding", path)
            return False
        for k in self._CACHE_KEYS:
            setattr(self, k, z[k])
        log.info("loaded transitions from %s", path)
        return True

    # ------------------------------------------------------------------ seeds
    def lin(self, ijk):
        return np.ravel_multi_index(tuple(np.asarray(ijk, int).T + 1), self.pshape)

    def seed_peaks(self, vlin):
        """fODF peak vertices of a (padded, linear) voxel; empty if it has no fODF."""
        xr = self.xrow[vlin]
        return np.zeros(0, np.int64) if xr < 0 else seed_peaks(self.X[xr], self.geo)

    def launch_halves(self, seed_vlin, dptr, dirs, out, dead):
        """First exits of half-walks; out (H, 26, nb, ncomp)."""
        o = np.empty(out.shape[:3])
        K.launch_batch_tree(seed_vlin, dptr, dirs, *self._kargs_tree(), self.nb, o, dead,
                            float(self.cfg.stop_mass))
        out[..., 0] = o

    def release_pmfs(self):
        """The bundle solves need only the transitions; on large grids the fODFs are big."""
        self.X = self.Y = None

    # ------------------------------------------------------------------ operators
    def mul_forward(self, x, out):
        """out = Q^T x (walker mass one step on)."""
        K.stencil_forward(self.Q_ptr, self.Q_slot, self.Q_w, self.tmpl_n, self.tmpl_b, self.nbr_t,
                          self.nb, x, out, self.chunk_ptr)
        return out

    def mul_backward(self, x, out):
        """out = Q x (completion probabilities one step back)."""
        K.stencil_backward(self.Q_ptr, self.Q_slot, self.Q_w, self.tmpl_n, self.tmpl_b, self.nbr_t,
                           self.nb, x, out)
        return out

    def absorb_rhs(self, absw):
        """(n_states, C): probability of being absorbed next step into voxels weighted by absw."""
        import scipy.sparse as sps

        R = sps.csr_matrix((self.R_val, self.R_ind, self.R_ptr), shape=(self.n_states, self.a_lin.size))
        return np.asarray(R @ absw)

    def absorbed_from(self, S):
        """(Na, C) absorbed mass from summed occupancy S."""
        out = np.empty((self.a_lin.size, S.shape[1]))
        K.csr_tmatmat(self.R_ptr, self.R_ind, self.R_val, np.ascontiguousarray(S), out)
        return out

    def to_volume(self, values, which="transient"):
        vol = np.zeros(int(np.prod(self.pshape)), np.float32)
        vol[self.t_lin if which == "transient" else self.a_lin] = values
        return vol.reshape(self.pshape)[1:-1, 1:-1, 1:-1]
