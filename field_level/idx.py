# !/usr/bin/env python3

import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial

from PT_field import coord

def idx_func(ng,a,b,c):
    ngo2 = ng // 2
    if c > ngo2:
        c = ng - c
    return int((a*ng+b)*(ngo2+1)+c)

def indep_coord(ng):
    ng2 = ng*ng
    ng3 = ng*ng*ng
    ngo2 = ng // 2
    ng2o2 = ng2 // 2
    ng3o2 = ng3 // 2
    # num = ng * ng * (ng//2 + 1)  # actual size
    
    ### c = 0 plane
    coord_czero_real = np.array([0, 
                                 idx_func(ng,ngo2,0,0), 
                                 idx_func(ng,0,ngo2,0), 
                                 idx_func(ng,ngo2,ngo2,0)
                                 ], dtype=np.uint32)
    
    coord_czero_azero_bhalf = np.arange(idx_func(ng,0,1,0), 
                                        idx_func(ng,0,ngo2-1,0)+1, 
                                        ngo2+1, 
                                        dtype=np.uint32)
    
    coord_czero_azero_bhalf_conj = ng2o2 + ng - coord_czero_azero_bhalf

    coord_czero_ahalf_bzero = np.arange(idx_func(ng,1,0,0), 
                                        idx_func(ng,ngo2-1,0,0)+1, 
                                        ng2o2+ng, 
                                        dtype=np.uint32)
    
    coord_czero_ahalf_bzero_conj = ng3o2 + ng2 - coord_czero_ahalf_bzero

    coord_czero_ahalf_bngo2 = coord_czero_ahalf_bzero + (ng2o2 // 2) + ngo2
    coord_czero_ahalf_bngo2_conj = coord_czero_ahalf_bzero_conj + (ng2o2 // 2) + ngo2
    
    coord_czero_aall_bhalf = np.array([
        np.arange(
            idx_func(ng,1,i+1,0), 
            idx_func(ng,ng-1,i+1,0)+1, 
            ng2o2+ng
            ) for i in range(ngo2-1)
            ], dtype=np.uint32)
    coord_czero_aall_bhalf_conj = ng3o2 + 3*ng2o2 + ng - coord_czero_aall_bhalf

    ### c = ng/2 plane
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
            coord_cngo2_aall_bhalf_conj,)

def indep_coord_stack(ng):
    (coord_czero_real, coord_czero_azero_bhalf, coord_czero_azero_bhalf_conj,
     coord_czero_ahalf_bzero, coord_czero_ahalf_bzero_conj,
     coord_czero_ahalf_bngo2, coord_czero_ahalf_bngo2_conj,
     coord_czero_aall_bhalf, coord_czero_aall_bhalf_conj,
     coord_cngo2_real, coord_cngo2_azero_bhalf, coord_cngo2_azero_bhalf_conj,
     coord_cngo2_ahalf_bzero, coord_cngo2_ahalf_bzero_conj,
     coord_cngo2_ahalf_bngo2, coord_cngo2_ahalf_bngo2_conj,
     coord_cngo2_aall_bhalf, coord_cngo2_aall_bhalf_conj) = indep_coord(ng)
    
    idx_conjugate_re = np.hstack([0,
                                   coord_czero_azero_bhalf_conj,
                                   coord_czero_ahalf_bzero_conj,
                                   coord_czero_ahalf_bngo2_conj,
                                   coord_czero_aall_bhalf_conj.ravel(),
                                   coord_cngo2_azero_bhalf_conj,
                                   coord_cngo2_ahalf_bzero_conj,
                                   coord_cngo2_ahalf_bngo2_conj,
                                   coord_cngo2_aall_bhalf_conj.ravel()])

    idx_conjugate_im = np.hstack([coord_czero_real,
                                   coord_cngo2_real,
                                   coord_czero_azero_bhalf_conj,
                                   coord_czero_ahalf_bzero_conj,
                                   coord_czero_ahalf_bngo2_conj,
                                   coord_czero_aall_bhalf_conj.ravel(),
                                   coord_cngo2_azero_bhalf_conj,
                                   coord_cngo2_ahalf_bzero_conj,
                                   coord_cngo2_ahalf_bngo2_conj,
                                   coord_cngo2_aall_bhalf_conj.ravel()])
    
    return jnp.asarray(idx_conjugate_re, dtype=jnp.int32), jnp.asarray(idx_conjugate_im, dtype=jnp.int32)

def _build_keep_indices(num: int, idx_delete) -> jnp.ndarray:
    """Return keep indices as int32 for jnp.take."""
    # Use JAX so the result can be a device constant closed over by a jitted fn
    mask = jnp.ones((num,), dtype=bool)
    mask = mask.at[jnp.array(idx_delete, dtype=jnp.int32)].set(False)
    keep = jnp.nonzero(mask, size=None)[0]
    return keep.astype(jnp.int32)

def build_indep_taker(ng: int,
                      idx_conjugate_real,
                      idx_conjugate_imag):
    """
    Precompute 'keep' indices once and return a gather-based extractor:
      taker(fieldk) -> stacked [Re_keep, Im_keep]
    """
    num = ng * ng * (ng // 2 + 1)
    keep_re = _build_keep_indices(num, idx_conjugate_real)
    keep_im = _build_keep_indices(num, idx_conjugate_imag)

    @jit
    def taker(fieldk: jnp.ndarray) -> jnp.ndarray:
        # fieldk: (ng, ng, ng//2+1) complex
        f1d = fieldk.reshape(-1)
        re = jnp.take(f1d.real, keep_re, mode='clip')
        im = jnp.take(f1d.imag, keep_im, mode='clip')
        return jnp.concatenate([re, im], axis=0)

    return taker, keep_re, keep_im

def _delete_mask(num: int, del_idx) -> jnp.ndarray:
    """Return a boolean mask where True means 'to be deleted'."""
    mask = jnp.zeros((num,), dtype=bool)
    return mask.at[jnp.asarray(del_idx, dtype=jnp.int32)].set(True)

def build_indep_taker_with_keep(ng: int,
                                keep_re: jnp.ndarray,
                                keep_im: jnp.ndarray):
    """
    Return a gather-based extractor with precomputed keep indices.

    keep_re / keep_im must be 1D int32 arrays of indices into the flattened rfftn layout.
    """
    keep_re = jnp.asarray(keep_re, dtype=jnp.int32)
    keep_im = jnp.asarray(keep_im, dtype=jnp.int32)

    @jit
    def taker(fieldk: jnp.ndarray) -> jnp.ndarray:
        # fieldk: (ng, ng, ng//2+1) complex
        f1d = fieldk.reshape(-1)
        re = jnp.take(f1d.real, keep_re, mode='clip')
        im = jnp.take(f1d.imag, keep_im, mode='clip')
        return jnp.concatenate([re, im], axis=0)

    return taker, keep_re, keep_im

def build_keep_from_delete_and_kmax(ng: int,
                                    boxsize: float,
                                    kmax: float,
                                    del_re,
                                    del_im,
                                    dtype=jnp.float32):
    """
    Build keep indices for independent modes under a spherical cut (k^2 <= kmax^2).

    We intersect:
      - below-k mask (spherical)
      - complement of conjugate-delete mask (from indep_coord_stack)
    """
    # number of complex rfftn modes
    num = ng * ng * (ng // 2 + 1)

    # build k^2 on rfftn grid (3D), then flatten
    kx, ky, kz = coord.kaxes_1d(ng, boxsize, dtype=dtype)
    kx2 = kx**2
    ky2 = ky**2
    kz2 = kz**2
    k2 = (kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]).reshape((num,))
    below = (k2 <= (kmax * kmax))

    # delete masks for conjugates (True=delete), then complement to keep
    del_mask_re = _delete_mask(num, del_re)
    del_mask_im = _delete_mask(num, del_im)
    keep_mask_re = (~del_mask_re) & below
    keep_mask_im = (~del_mask_im) & below

    keep_re = jnp.nonzero(keep_mask_re, size=None)[0].astype(jnp.int32)
    keep_im = jnp.nonzero(keep_mask_im, size=None)[0].astype(jnp.int32)
    return keep_re, keep_im

def _gauss_plan(ng: int):
    """Precompute index tuples once on host; returned arrays are NumPy int32."""
    return indep_coord(ng)

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
        # same math as your current gauss_1d_to_3d, but indep_coord(ng) を呼ばない
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
