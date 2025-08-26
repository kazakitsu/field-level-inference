# !/usr/bin/env python3
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

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
    This is needed by build_gauss_1d_to_3d and kept unchanged.
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

# --- private helpers for keep-mask based extraction ---
def _unflatten_abc(ng: int):
    """Return flattened (a,b,c) coordinates for rfftn layout of shape (ng, ng, ng//2+1)."""
    ngo2 = ng // 2
    a = jnp.arange(ng, dtype=jnp.int32)[:, None, None]
    b = jnp.arange(ng, dtype=jnp.int32)[None, :, None]
    c = jnp.arange(ngo2 + 1, dtype=jnp.int32)[None, None, :]
    A = jnp.broadcast_to(a, (ng, ng, ngo2 + 1))
    B = jnp.broadcast_to(b, (ng, ng, ngo2 + 1))
    C = jnp.broadcast_to(c, (ng, ng, ngo2 + 1))
    return A.reshape(-1), B.reshape(-1), C.reshape(-1)

def _conj_flat_indices(ng: int):
    """
    For each flattened rfftn index, return the flattened index of its Hermitian conjugate.
    """
    A, B, C = _unflatten_abc(ng)  # (num,)
    ng_i32 = jnp.int32(ng)
    Aconj = (ng_i32 - A) % ng_i32
    Bconj = (ng_i32 - B) % ng_i32
    Cconj = C  # c stays within [0..ng/2]
    idx = ((A * ng_i32 + B) * (ng_i32 // 2 + 1) + C).astype(jnp.int32)
    jdx = ((Aconj * ng_i32 + Bconj) * (ng_i32 // 2 + 1) + Cconj).astype(jnp.int32)
    return idx, jdx

# --- keep builders (mask-based, recommended) ---
def build_keep_conjugate_only(ng: int):
    """
    No k-mask. Choose one representative per conjugate pair on the full grid.
    keep_re includes all representatives; keep_im excludes self-conjugate (pure-real) modes.
    """
    idx, jdx = _conj_flat_indices(ng)
    self_conj = (idx == jdx)
    pick_left = (idx < jdx)         # pick only one side of each pair
    keep_rep = pick_left | self_conj
    keep_re = jnp.nonzero(keep_rep, size=None)[0].astype(jnp.int32)
    keep_im = jnp.nonzero(keep_rep & (~self_conj), size=None)[0].astype(jnp.int32)
    return keep_re, keep_im

def build_keep_spherical(ng: int, boxsize: float, kmax: float, dtype=jnp.float32):
    """
    Spherical cut (|k| <= kmax) AND unique representative per conjugate pair.
    Self-conjugate (pure-real) modes are kept only in Re set.
    """
    # build |k|^2 on the rfftn layout
    kx, ky, kz = coord.kaxes_1d(ng, boxsize, dtype=dtype)   # (ng,), (ng,), (ng//2+1,)
    k2 = (kx**2)[:, None, None] + (ky**2)[None, :, None] + (kz**2)[None, None, :]
    num = ng * ng * (ng // 2 + 1)
    k2 = k2.reshape((num,))
    below = (k2 <= (kmax * kmax))  # True inside the sphere

    # conjugate pairs
    idx, jdx = _conj_flat_indices(ng)
    self_conj = (idx == jdx)
    pick_left  = (idx < jdx)
    pick_right = (~pick_left) & (~self_conj)

    # representative selection with spherical mask
    keep_rep = (below & pick_left) | (below[jdx] & pick_right) | (below & self_conj)

    keep_re = jnp.nonzero(keep_rep, size=None)[0].astype(jnp.int32)
    keep_im = jnp.nonzero(keep_rep & (~self_conj), size=None)[0].astype(jnp.int32)
    return keep_re, keep_im

def build_indep_taker_with_keep(ng: int, keep_re: jnp.ndarray, keep_im: jnp.ndarray):
    """
    Create a gather-based extractor: taker(fieldk) -> concat([Re_keep, Im_keep]).
    keep_re/keep_im are flattened rfftn indices (int32).
    """
    keep_re = jnp.asarray(keep_re, dtype=jnp.int32)
    keep_im = jnp.asarray(keep_im, dtype=jnp.int32)

    @jit
    def taker(fieldk: jnp.ndarray) -> jnp.ndarray:
        # fieldk: complex array of shape (ng, ng, ng//2+1) in rfftn layout
        f1d = fieldk.reshape(-1)
        re = jnp.take(f1d.real, keep_re, mode='clip')
        im = jnp.take(f1d.imag, keep_im, mode='clip')
        return jnp.concatenate([re, im], axis=0)

    return taker, keep_re, keep_im


@partial(jit, static_argnames=('ng',))
def gauss_1d_to_3d(gaussian_1d: jnp.ndarray, ng: int):
    """
    Map ng^3 real Gaussian DOF to a hermitian-symmetric rfftn layout (ng, ng, ng//2+1).
    """
    dtype = gaussian_1d.dtype
    ng2  = ng * ng
    ng3  = ng * ng * ng
    ngo2 = ng // 2
    ng2o2 = ng2 // 2
    ng3o2 = ng3 // 2
    num   = ng * ng * (ngo2 + 1)

    (coord_czero_real, coord_czero_azero_bhalf, coord_czero_azero_bhalf_conj,
     coord_czero_ahalf_bzero, coord_czero_ahalf_bzero_conj,
     coord_czero_ahalf_bngo2, coord_czero_ahalf_bngo2_conj,
     coord_czero_aall_bhalf, coord_czero_aall_bhalf_conj,
     coord_cngo2_real, coord_cngo2_azero_bhalf, coord_cngo2_azero_bhalf_conj,
     coord_cngo2_ahalf_bzero, coord_cngo2_ahalf_bzero_conj,
     coord_cngo2_ahalf_bngo2, coord_cngo2_ahalf_bngo2_conj,
     coord_cngo2_aall_bhalf, coord_cngo2_aall_bhalf_conj) = indep_coord(ng)  # traced once per ng

    # Ensure JAX arrays (int32)
    coord_czero_real = jnp.asarray(coord_czero_real, dtype=jnp.int32)
    coord_czero_azero_bhalf = jnp.asarray(coord_czero_azero_bhalf, dtype=jnp.int32)
    coord_czero_azero_bhalf_conj = jnp.asarray(coord_czero_azero_bhalf_conj, dtype=jnp.int32)
    coord_czero_ahalf_bzero = jnp.asarray(coord_czero_ahalf_bzero, dtype=jnp.int32)
    coord_czero_ahalf_bzero_conj = jnp.asarray(coord_czero_ahalf_bzero_conj, dtype=jnp.int32)
    coord_czero_ahalf_bngo2 = jnp.asarray(coord_czero_ahalf_bngo2, dtype=jnp.int32)
    coord_czero_ahalf_bngo2_conj = jnp.asarray(coord_czero_ahalf_bngo2_conj, dtype=jnp.int32)
    coord_czero_aall_bhalf = jnp.asarray(coord_czero_aall_bhalf, dtype=jnp.int32).ravel()
    coord_czero_aall_bhalf_conj = jnp.asarray(coord_czero_aall_bhalf_conj, dtype=jnp.int32).ravel()
    coord_cngo2_real = jnp.asarray(coord_cngo2_real, dtype=jnp.int32)
    coord_cngo2_azero_bhalf = jnp.asarray(coord_cngo2_azero_bhalf, dtype=jnp.int32)
    coord_cngo2_azero_bhalf_conj = jnp.asarray(coord_cngo2_azero_bhalf_conj, dtype=jnp.int32)
    coord_cngo2_ahalf_bzero = jnp.asarray(coord_cngo2_ahalf_bzero, dtype=jnp.int32)
    coord_cngo2_ahalf_bzero_conj = jnp.asarray(coord_cngo2_ahalf_bzero_conj, dtype=jnp.int32)
    coord_cngo2_ahalf_bngo2 = jnp.asarray(coord_cngo2_ahalf_bngo2, dtype=jnp.int32)
    coord_cngo2_ahalf_bngo2_conj = jnp.asarray(coord_cngo2_ahalf_bngo2_conj, dtype=jnp.int32)
    coord_cngo2_aall_bhalf = jnp.asarray(coord_cngo2_aall_bhalf, dtype=jnp.int32).ravel()
    coord_cngo2_aall_bhalf_conj = jnp.asarray(coord_cngo2_aall_bhalf_conj, dtype=jnp.int32).ravel()
    
    fieldk_re_1d = jnp.zeros(num)
    fieldk_im_1d = jnp.zeros(num)
    
    ### c=0 plane
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

    fieldk_re_1d = fieldk_re_1d.at[coord_czero_aall_bhalf.ravel()].set(gaussian_1d[3*ng-2:ng2o2+3*ngo2-1])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_aall_bhalf.ravel()].set(gaussian_1d[ng2o2+3*ngo2-1:ng2])

    fieldk_re_1d = fieldk_re_1d.at[coord_czero_aall_bhalf_conj.ravel()].set(fieldk_re_1d[coord_czero_aall_bhalf.ravel()])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_aall_bhalf_conj.ravel()].set(-fieldk_im_1d[coord_czero_aall_bhalf.ravel()])
    
    ### c=ng//2 plane
    
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

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_aall_bhalf.ravel()].set(gaussian_1d[ng2+3*ng-2:ng2+ng2o2+3*ngo2-1])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_aall_bhalf.ravel()].set(gaussian_1d[ng2+ng2o2+3*ngo2-1:2*ng2])

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_aall_bhalf_conj.ravel()].set(fieldk_re_1d[coord_cngo2_aall_bhalf.ravel()])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_aall_bhalf_conj.ravel()].set(-fieldk_im_1d[coord_cngo2_aall_bhalf.ravel()])
    
    
    fieldk_re_1d = fieldk_re_1d.reshape(ng,ng,ngo2+1).transpose((2,1,0)).reshape(num)
    fieldk_re_1d = fieldk_re_1d.at[ng2:ng3o2].set(gaussian_1d[2*ng2:ng3o2+ng2])
    fieldk_re_1d = fieldk_re_1d.reshape(ngo2+1,ng,ng).transpose((2,1,0)).reshape(num)

    fieldk_im_1d = fieldk_im_1d.reshape(ng,ng,ngo2+1).transpose((2,1,0)).reshape(num)
    fieldk_im_1d = fieldk_im_1d.at[ng2:ng3o2].set(gaussian_1d[ng3o2+ng2:ng3])
    fieldk_im_1d = fieldk_im_1d.reshape(ngo2+1,ng,ng).transpose((2,1,0)).reshape(num)

    return fieldk_re_1d.reshape(ng,ng,ngo2+1) + 1j*fieldk_im_1d.reshape(ng,ng,ngo2+1)

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
