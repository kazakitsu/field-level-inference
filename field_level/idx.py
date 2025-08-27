# !/usr/bin/env python3
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

import logging
logger = logging.getLogger(__name__)

import PT_field.coord_jax as coord

# --- kept for gauss_1d_to_3d plan (used elsewhere) ---
def idx_func(ng, a, b, c):
    ngo2 = ng // 2
    if c > ngo2:
        c = ng - c
    return int((a * ng + b) * (ngo2 + 1) + c)

def indep_coord(ng):
    """
    Return index plan for mapping 1D Gaussian DOF to Hermitian-symmetric rfftn layout.
    """
    ng2 = ng * ng
    ng3 = ng * ng * ng
    ngo2 = ng // 2
    ng2o2 = ng2 // 2
    ng3o2 = ng3 // 2

    coord_czero_real = np.array([0,
                                 idx_func(ng, ngo2, 0, 0),
                                 idx_func(ng, 0, ngo2, 0),
                                 idx_func(ng, ngo2, ngo2, 0)], dtype=np.uint32)

    coord_czero_azero_bhalf = np.arange(idx_func(ng, 0, 1, 0),
                                        idx_func(ng, 0, ngo2 - 1, 0) + 1,
                                        ngo2 + 1, dtype=np.uint32)
    coord_czero_azero_bhalf_conj = ng2 // 2 + ng - coord_czero_azero_bhalf

    coord_czero_ahalf_bzero = np.arange(idx_func(ng, 1, 0, 0),
                                        idx_func(ng, ngo2 - 1, 0, 0) + 1,
                                        ng2 // 2 + ng, dtype=np.uint32)
    coord_czero_ahalf_bzero_conj = ng3 // 2 + ng2 - coord_czero_ahalf_bzero

    coord_czero_ahalf_bngo2 = coord_czero_ahalf_bzero + (ng2 // 4) + ngo2
    coord_czero_ahalf_bngo2_conj = coord_czero_ahalf_bzero_conj + (ng2 // 4) + ngo2

    coord_czero_aall_bhalf = np.array([
        np.arange(idx_func(ng, 1, i + 1, 0),
                  idx_func(ng, ng - 1, i + 1, 0) + 1,
                  ng2 // 2 + ng) for i in range(ngo2 - 1)
    ], dtype=np.uint32)
    coord_czero_aall_bhalf_conj = ng3 // 2 + 3 * (ng2 // 2) + ng - coord_czero_aall_bhalf

    coord_cngo2_real = coord_czero_real + ngo2
    coord_cngo2_azero_bhalf = ngo2 + coord_czero_azero_bhalf
    coord_cngo2_azero_bhalf_conj = ngo2 + coord_czero_azero_bhalf_conj
    coord_cngo2_ahalf_bzero = ngo2 + coord_czero_ahalf_bzero
    coord_cngo2_ahalf_bzero_conj = ngo2 + coord_czero_ahalf_bzero_conj
    coord_cngo2_ahalf_bngo2 = ngo2 + coord_czero_ahalf_bngo2
    coord_cngo2_ahalf_bngo2_conj = ngo2 + coord_czero_ahalf_bngo2_conj
    coord_cngo2_aall_bhalf = ngo2 + coord_czero_aall_bhalf
    coord_cngo2_aall_bhalf_conj = ngo2 + coord_czero_aall_bhalf_conj

    return (coord_czero_real,
            coord_czero_azero_bhalf,
            coord_czero_azero_bhalf_conj,
            coord_czero_ahalf_bzero,
            coord_czero_ahalf_bzero_conj,
            coord_czero_ahalf_bngo2,
            coord_czero_ahalf_bngo2_conj,
            coord_czero_aall_bhalf,
            coord_czero_aall_bhalf_conj,
            coord_cngo2_real,
            coord_cngo2_azero_bhalf,
            coord_cngo2_azero_bhalf_conj,
            coord_cngo2_ahalf_bzero,
            coord_cngo2_ahalf_bzero_conj,
            coord_cngo2_ahalf_bngo2,
            coord_cngo2_ahalf_bngo2_conj,
            coord_cngo2_aall_bhalf,
            coord_cngo2_aall_bhalf_conj)

def _keep_masks_half_rfftn(ng: int):
    """
    Build boolean keep masks (flattened) for Re and Im parts on an rfftn half-spectrum.
    Rules:
      - kz in (1 .. ng//2-1): keep all (kx, ky) for both Re and Im
      - kz in {0, ng/2 if even}: pick 2D Hermitian half-plane
          * keep ky in 1..ng//2-1  -> keep all kx
          * keep ky in {0, ng/2}   -> keep kx in 0..ng//2
          * on self-conjugate points (kx in {0, ng/2}, ky in {0, ng/2} on boundary): Im is False
    """
    ng = int(ng)
    ngo2 = ng // 2
    has_nyq = (ng % 2 == 0)

    ix = jnp.arange(ng)[:, None, None]
    iy = jnp.arange(ng)[None, :, None]
    iz = jnp.arange(ngo2 + 1)[None, None, :]

    # z-boundary planes where 2D Hermitian symmetry lives
    boundary = (iz == 0) | (has_nyq & (iz == ngo2))

    # y half-plane selection on z-boundary:
    ky_lt_half = (iy > 0) & (iy < ngo2)            # ky in 1..ngo2-1
    ky_tie     = (iy == 0) | (has_nyq & (iy == ngo2))
    kx_half    = (ix <= ngo2)                      # kx in 0..ngo2

    keep_plane = jnp.where(boundary, ky_lt_half | (ky_tie & kx_half), True)

    keep_re = keep_plane
    keep_im = keep_plane

    # On boundary planes, four self-conjugate points have purely real values
    if has_nyq:
        kx_self = (ix == 0) | (ix == ngo2)
        ky_self = (iy == 0) | (iy == ngo2)
    else:
        kx_self = (ix == 0)
        ky_self = (iy == 0)
    self_points = boundary & kx_self & ky_self
    keep_im = keep_im & (~self_points)

    return keep_re.reshape(-1), keep_im.reshape(-1)

def _apply_kmax_mask(keep_re_flat, keep_im_flat, kx, ky, kz, kmax: float):
    """
    Intersect with |k|<=kmax. kx,ky,kz are 1D physical wave-number axes.
    """
    kx = jnp.asarray(kx); ky = jnp.asarray(ky); kz = jnp.asarray(kz)
    k2 = (kx[:, None, None]**2 +
          ky[None, :, None]**2 +
          kz[None, None, :]**2)
    below = (k2 <= (kmax * kmax)).reshape(-1)
    keep_re_flat = keep_re_flat & below
    keep_im_flat = keep_im_flat & below
    return keep_re_flat, keep_im_flat

def indep_modes_kmax_indices(ng: int,
                                *,
                                kx=None, ky=None, kz=None,
                                kmax: float | None = None):
    """
    Public helper:
      - Build flattened boolean masks keep_re, keep_im
      - Return gather indices pos_re, pos_im as int32
      - If 0<kmax<=1.0, apply the |k|<=kmax mask using provided 1D axes
      - If kmax is None, no k-cut is applied
    """
    keep_re, keep_im = _keep_masks_half_rfftn(ng)

    if (kmax is not None) and (kmax > 0.0) and (kmax <= 1.0):
        assert kx is not None and ky is not None and kz is not None, \
            "kx, ky, kz are required when applying a physical kmax cut."
        keep_re, keep_im = _apply_kmax_mask(keep_re, keep_im, kx, ky, kz, float(kmax))

    idx_re = jnp.where(keep_re, size=keep_re.sum(), fill_value=0)[0].astype(jnp.int32)
    idx_im = jnp.where(keep_im, size=keep_im.sum(), fill_value=0)[0].astype(jnp.int32)

    return idx_re, idx_im

def build_gauss_1d_to_3d(ng: int):
    """
    Precompute index plan once and return a jitted function:
       fn = build_gauss_1d_to_3d(ng)
       fieldk = fn(gaussian_1d)
    """
    # precompute all index arrays once on host, then convert to JAX int32
    plan_np = indep_coord(ng)
    (coord_czero_real, coord_czero_azero_bhalf, coord_czero_azero_bhalf_conj,
     coord_czero_ahalf_bzero, coord_czero_ahalf_bzero_conj,
     coord_czero_ahalf_bngo2, coord_czero_ahalf_bngo2_conj,
     coord_czero_aall_bhalf, coord_czero_aall_bhalf_conj,
     coord_cngo2_real, coord_cngo2_azero_bhalf, coord_cngo2_azero_bhalf_conj,
     coord_cngo2_ahalf_bzero, coord_cngo2_ahalf_bzero_conj,
     coord_cngo2_ahalf_bngo2, coord_cngo2_ahalf_bngo2_conj,
     coord_cngo2_aall_bhalf, coord_cngo2_aall_bhalf_conj) = [
        jnp.asarray(a, dtype=jnp.int32) for a in plan_np
    ]

    @jit
    def _fn(gaussian_1d: jnp.ndarray) -> jnp.ndarray:
        ng2  = ng * ng
        ng3  = ng * ng * ng
        ngo2 = ng // 2
        ng2o2 = ng2 // 2
        ng3o2 = ng3 // 2
        num   = ng * ng * (ngo2 + 1)

        fieldk_re_1d = jnp.zeros((num,), dtype=gaussian_1d.dtype)
        fieldk_im_1d = jnp.zeros((num,), dtype=gaussian_1d.dtype)

        # ---- c = 0 plane
        fieldk_re_1d = fieldk_re_1d.at[coord_czero_real].set(gaussian_1d[0:4])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_real].set(0.)
        fieldk_re_1d = fieldk_re_1d.at[0].set(0.)

        fieldk_re_1d = fieldk_re_1d.at[coord_czero_azero_bhalf].set(gaussian_1d[4:ngo2+3])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_azero_bhalf].set(gaussian_1d[ngo2+3:ng+2])

        fieldk_re_1d = fieldk_re_1d.at[coord_czero_azero_bhalf_conj].set(fieldk_re_1d[coord_czero_azero_bhalf])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_azero_bhalf_conj].set(-fieldk_im_1d[coord_czero_azero_bhalf])

        fieldk_re_1d = fieldk_re_1d.at[coord_czero_ahalf_bzero].set(gaussian_1d[ng+2:ng+ngo2+1])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_ahalf_bzero].set(gaussian_1d[ng+ngo2+1:2*ng])

        fieldk_re_1d = fieldk_re_1d.at[coord_czero_ahalf_bzero_conj].set(fieldk_re_1d[coord_czero_ahalf_bzero])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_ahalf_bzero_conj].set(-fieldk_im_1d[coord_czero_ahalf_bzero])

        fieldk_re_1d = fieldk_re_1d.at[coord_czero_ahalf_bngo2].set(gaussian_1d[2*ng:2*ng+ngo2-1])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_ahalf_bngo2].set(gaussian_1d[2*ng+ngo2-1:3*ng-2])

        fieldk_re_1d = fieldk_re_1d.at[coord_czero_ahalf_bngo2_conj].set(fieldk_re_1d[coord_czero_ahalf_bngo2])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_ahalf_bngo2_conj].set(-fieldk_im_1d[coord_czero_ahalf_bngo2])

        coord_czero_aall_bhalf_flat = coord_czero_aall_bhalf.ravel()
        coord_czero_aall_bhalf_conj_flat = coord_czero_aall_bhalf_conj.ravel()
        fieldk_re_1d = fieldk_re_1d.at[coord_czero_aall_bhalf_flat].set(gaussian_1d[3*ng-2:ng2o2+3*ngo2-1])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_aall_bhalf_flat].set(gaussian_1d[ng2o2+3*ngo2-1:ng2])
        fieldk_re_1d = fieldk_re_1d.at[coord_czero_aall_bhalf_conj_flat].set(fieldk_re_1d[coord_czero_aall_bhalf_flat])
        fieldk_im_1d = fieldk_im_1d.at[coord_czero_aall_bhalf_conj_flat].set(-fieldk_im_1d[coord_czero_aall_bhalf_flat])

        # ---- c = ng//2 plane
        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_real].set(gaussian_1d[ng2:ng2+4])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_real].set(0.)

        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_azero_bhalf].set(gaussian_1d[ng2+4:ng2+ngo2+3])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_azero_bhalf].set(gaussian_1d[ng2+ngo2+3:ng2+ng+2])

        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_azero_bhalf_conj].set(fieldk_re_1d[coord_cngo2_azero_bhalf])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_azero_bhalf_conj].set(-fieldk_im_1d[coord_cngo2_azero_bhalf])

        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_ahalf_bzero].set(gaussian_1d[ng2+ng+2:ng2+ng+ngo2+1])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_ahalf_bzero].set(gaussian_1d[ng2+ng+ngo2+1:ng2+2*ng])

        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_ahalf_bzero_conj].set(fieldk_re_1d[coord_cngo2_ahalf_bzero])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_ahalf_bzero_conj].set(-fieldk_im_1d[coord_cngo2_ahalf_bzero])

        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_ahalf_bngo2].set(gaussian_1d[ng2+2*ng:ng2+2*ng+ngo2-1])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_ahalf_bngo2].set(gaussian_1d[ng2+2*ng+ngo2-1:ng2+3*ng-2])

        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_ahalf_bngo2_conj].set(fieldk_re_1d[coord_cngo2_ahalf_bngo2])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_ahalf_bngo2_conj].set(-fieldk_im_1d[coord_cngo2_ahalf_bngo2])

        coord_cngo2_aall_bhalf_flat = coord_cngo2_aall_bhalf.ravel()
        coord_cngo2_aall_bhalf_conj_flat = coord_cngo2_aall_bhalf_conj.ravel()
        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_aall_bhalf_flat].set(gaussian_1d[ng2+3*ng-2:ng2+ng2o2+3*ngo2-1])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_aall_bhalf_flat].set(gaussian_1d[ng2+ng2o2+3*ngo2-1:2*ng2])
        fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_aall_bhalf_conj_flat].set(fieldk_re_1d[coord_cngo2_aall_bhalf_flat])
        fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_aall_bhalf_conj_flat].set(-fieldk_im_1d[coord_cngo2_aall_bhalf_flat])

        # middle slab (no special symmetry points)
        fieldk_re_1d = fieldk_re_1d.reshape(ng, ng, ngo2+1).transpose((2, 1, 0)).reshape(num)
        fieldk_re_1d = fieldk_re_1d.at[ng2:ng3o2].set(gaussian_1d[2*ng2:ng3o2+ng2])
        fieldk_re_1d = fieldk_re_1d.reshape(ngo2+1, ng, ng).transpose((2, 1, 0)).reshape(num)

        fieldk_im_1d = fieldk_im_1d.reshape(ng, ng, ngo2+1).transpose((2, 1, 0)).reshape(num)
        fieldk_im_1d = fieldk_im_1d.at[ng2:ng3o2].set(gaussian_1d[ng3o2+ng2:ng3])
        fieldk_im_1d = fieldk_im_1d.reshape(ngo2+1, ng, ng).transpose((2, 1, 0)).reshape(num)

        return fieldk_re_1d.reshape(ng, ng, ngo2+1) + 1j * fieldk_im_1d.reshape(ng, ng, ngo2+1)

    return _fn
