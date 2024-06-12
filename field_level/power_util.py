#!/usr/bin/env python3

import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

jax.config.update('jax_enable_x64', True)

import field_level.coord as coord

def power_compute(fieldk_1, fieldk_2, boxsize, nbin=60, kmin=0.0, kmax=0.6, ell=0, leg_fac=True):
    ng = fieldk_1.shape[0]
    if fieldk_1.shape != fieldk_2.shape:
        raise ValueError(f'{fieldk_1.shape} != {fieldk_2.shape}')

    kf = 2*jnp.pi/boxsize
    kvec = coord.rfftn_kvec([ng,ng,ng], boxsize)
    k2 = (kvec ** 2).sum(axis=0)
    k = jnp.sqrt(k2)

    vol = boxsize**3

    k_arr = jnp.linspace(kmin, kmax, nbin+1)

    ### Computing powers
    if leg_fac == True:
        legendre_fac = 2.0*ell + 1.0
    else:
        legendre_fac = 1.0
    Pk = fieldk_1 * fieldk_2.conj()
    
    ### Set the appropriate Nk for each k
    Nk = jnp.full_like(Pk, 2, dtype=jnp.int32)
    Nk = Nk.at[..., 0].set(1)
    if fieldk_1.shape[-1] % 2 == 0:
        Nk = Nk.at[..., -1].set(1)
    
    ### Multopoles
    if ell > 0:
        mu2 = kvec[2]*kvec[2]/k2
        mu2 = mu2.at[0, 0, 0].set(0.0)
    if ell == 2:
        mu_fac = legendre_fac * 0.5 * (3.0*mu2 - 1.0)
        Pk *= mu_fac
    elif ell == 4:
        mu_fac = legendre_fac * 0.125 * (35.0*mu2*mu2 - 30.0*mu2 + 3.0)
        Pk *= mu_fac
    elif ell == 6:
        mu_fac = legendre_fac * 0.0625 * (231.*mu2*mu2*mu2 - 315.*mu2*mu2 + 105.*mu2 - 5.)
        Pk *= mu_fac
    elif ell == 8:
        mu_fac = legendre_fac * 0.0078125 * (6435.*mu2*mu2*mu2*mu2  - 12012.*mu2*mu2*mu2 + 6930.*mu2*mu2  - 1260.*mu2 + 35.)
        Pk *= mu_fac
    elif ell == 10:
        mu_fac = legendre_fac * 0.00390625 * (46189.*mu2*mu2*mu2*mu2*mu2 - 109395.*mu2*mu2*mu2*mu2 + 90090.*mu2*mu2*mu2 - 30030.*mu2*mu2 + 3465.*mu2 - 63.)
        Pk *= mu_fac
    Pk = Pk.at[0,0,0].set(0.0)
    
    ### to 1D
    k = k.ravel()
    Pk = Pk.ravel()
    Nk = Nk.ravel()
    
    kidx = jnp.digitize(k, k_arr, right=True)
    
    k *= Nk
    Pk *= Nk
    
    k  = jnp.bincount(kidx, weights=k,  length=nbin+1)
    Pk = jnp.bincount(kidx, weights=Pk, length=nbin+1)
    Nk = jnp.bincount(kidx, weights=Nk, length=nbin+1)

    k  = k[1:nbin+1]
    Pk = Pk[1:nbin+1]
    Nk = Nk[1:nbin+1]
    k_arr = k_arr[:nbin+1]
    
    k /= Nk
    Pk /= Nk
    
    Pk *= vol

    return k, Pk, Nk

def window_legendre_compute(fieldk, boxsize, nbin=60, kmin=0.0, kmax=0.6, ellmax=6):
    ng = fieldk.shape[0]
    nell = int(ellmax/2) + 1

    kvec = coord.rfftn_kvec([ng,ng,ng], boxsize)
    k2 = (kvec ** 2).sum(axis=0)
    k_base = jnp.sqrt(k2)

    k_arr = jnp.linspace(kmin, kmax, nbin+1)
    
    ### Set the appropriate Nk for each k
    Nk_base = jnp.full_like(fieldk, 2, dtype=jnp.int32)
    Nk_base = Nk_base.at[..., 0].set(1)
    if fieldk.shape[-1] % 2 == 0:
        Nk_base = Nk_base.at[..., -1].set(1)
    
    ### Multopoles
    mu2 = kvec[2]*kvec[2]/k2
    mu2 = mu2.at[0, 0, 0].set(0.0)
    
    L_ells = jnp.ones((nell, fieldk.shape[0], fieldk.shape[1], fieldk.shape[2]))
    L_ells = L_ells.at[1].set( 0.5   * (3.0*mu2 - 1.0) )
    L_ells = L_ells.at[2].set( 0.125 * (35.0*mu2*mu2 - 30.0*mu2 + 3.0) )
    if ellmax > 4:
        L_ells = L_ells.at[3].set( 0.0625    * (231.*mu2*mu2*mu2 - 315.*mu2*mu2 + 105.*mu2 - 5.) )   ### ell = 6
    if ellmax > 5:
        L_ells = L_ells.at[4].set( 0.0078125 * (6435.*mu2*mu2*mu2*mu2  - 12012.*mu2*mu2*mu2 + 6930.*mu2*mu2  - 1260.*mu2 + 35.) ) ### ell = 8
    if ellmax > 8:
        L_ells = L_ells.at[5].set( 0.00390625* (46189.*mu2*mu2*mu2*mu2*mu2 - 109395.*mu2*mu2*mu2*mu2 + 90090.*mu2*mu2*mu2 - 30030.*mu2*mu2 + 3465.*mu2 - 63.) ) ### ell = 10
            
    ### compuute the window
    window = jnp.zeros((nell, nell, nbin))
    for ell1 in range(nell):
        legendre_fac = 2*2*ell1+1
        #legendre_fac = 1.
        for ell2 in range(nell):
            Pk = jnp.full_like(fieldk, 1., dtype=jnp.float64)
            Pk *= legendre_fac*L_ells[ell1]*L_ells[ell2]
            Pk = Pk.at[0,0,0].set(0.0)
                        
            k = k_base.flatten()
            Nk = Nk_base.flatten()
            Pk = Pk.ravel()
            
            kidx = jnp.digitize(k, k_arr, right=True)
            
            k *= Nk
            Pk *= Nk
    
            k  = jnp.bincount(kidx, weights=k,  length=nbin+1)
            Pk = jnp.bincount(kidx, weights=Pk, length=nbin+1)
            Nk = jnp.bincount(kidx, weights=Nk, length=nbin+1)
            
            k  = k[1:nbin+1]
            Pk = Pk[1:nbin+1]
            Nk = Nk[1:nbin+1]
            
            k /= Nk
            Pk /= Nk
    
            window = window.at[ell1, ell2].set(Pk)
            
    return window

def covariance_legendre_compute(pk_3d, boxsize, nbin=60, kmin=0.0, kmax=0.6, ellmax=4):
    ng = pk_3d.shape[0]
    nell = int(ellmax/2) + 1

    kvec = coord.rfftn_kvec([ng,ng,ng], boxsize)
    k2 = (kvec ** 2).sum(axis=0)
    k_base = jnp.sqrt(k2)

    k_arr = jnp.linspace(kmin, kmax, nbin+1)
    
    ### Set the appropriate Nk for each k
    Nk_base = jnp.full_like(pk_3d, 2, dtype=jnp.int64)
    Nk_base = Nk_base.at[..., 0].set(1)
    if pk_3d.shape[-1] % 2 == 0:
        Nk_base = Nk_base.at[..., -1].set(1)
    
    ### Multopoles
    mu2 = kvec[2]*kvec[2]/k2
    mu2 = mu2.at[0, 0, 0].set(0.0)
        
    L_ells = jnp.ones((nell, pk_3d.shape[0], pk_3d.shape[1], pk_3d.shape[2]))
    L_ells = L_ells.at[1].set( 0.5   * (3.0*mu2 - 1.0) )
    L_ells = L_ells.at[2].set( 0.125 * (35.0*mu2*mu2 - 30.0*mu2 + 3.0) )
    if ellmax > 4:
        L_ells = L_ells.at[3].set( 0.0625    * (231.*mu2*mu2*mu2 - 315.*mu2*mu2 + 105.*mu2 - 5.) )   ### ell = 6
    if ellmax > 5:
        L_ells = L_ells.at[4].set( 0.0078125 * (6435.*mu2*mu2*mu2*mu2  - 12012.*mu2*mu2*mu2 + 6930.*mu2*mu2  - 1260.*mu2 + 35.) ) ### ell = 8
    if ellmax > 8:
        L_ells = L_ells.at[5].set( 0.00390625* (46189.*mu2*mu2*mu2*mu2*mu2 - 109395.*mu2*mu2*mu2*mu2 + 90090.*mu2*mu2*mu2 - 30030.*mu2*mu2 + 3465.*mu2 - 63.) ) ### ell = 10
            
    ### compuute the covariance
    cov = jnp.zeros((nell*nbin, nell*nbin))
    for ell1 in range(nell):
        legendre_fac1 = 2*2*ell1+1
        for ell2 in range(nell):
            legendre_fac2 = 2*2*ell2+1
            Pk = pk_3d**2
            Pk *= legendre_fac1*legendre_fac2*L_ells[ell1]*L_ells[ell2]
            Pk = Pk.at[0,0,0].set(0.0)
                        
            k = k_base.ravel()
            Nk = Nk_base.ravel()
            Pk = Pk.ravel()
            
            kidx = jnp.digitize(k, k_arr, right=True)
            
            Pk *= Nk
    
            Pk = jnp.bincount(kidx, weights=Pk, length=nbin+1)
            Nk = jnp.bincount(kidx, weights=Nk, length=nbin+1)
            
            Pk = Pk[1:nbin+1]
            Nk = Nk[1:nbin+1]
            
            Pk /= Nk**2
            Pk *= 2.0
            
            for ii in range(nbin):
                cov = cov.at[ell1*nbin+ii, ell2*nbin+ii].set(Pk[ii])
            
    return cov

@partial(jit, static_argnums=(3, 4, 5, 6))
def power_weight_compute(pk_3d, k_3d, mu2_3d, nbin=60, kmin=0.0, kmax=0.6, ellmax=4):
    #pk_3d = jnp.interp(k_3d, pk[0], pk[1])
    nell = int(ellmax/2) + 1

    k_arr = jnp.linspace(kmin, kmax, nbin+1)
    
    ### Set the appropriate Nk for each k
    Nk_base = jnp.full_like(pk_3d, 2, dtype=jnp.int32)
    Nk_base = Nk_base.at[..., 0].set(1)
    if pk_3d.shape[-1] % 2 == 0:
        Nk_base = Nk_base.at[..., -1].set(1)
    
    ### Multopoles
        
    L_ells = jnp.ones((nell, pk_3d.shape[0], pk_3d.shape[1], pk_3d.shape[2]))
    if ellmax > 0:
        L_ells = L_ells.at[1].set( 0.5   * (3.0*mu2_3d - 1.0) )
    if ellmax > 2:
        L_ells = L_ells.at[2].set( 0.125 * (35.0*mu2_3d*mu2_3d - 30.0*mu2_3d + 3.0) )
    if ellmax > 4:
        L_ells = L_ells.at[3].set( 0.0625    * (231.*mu2_3d*mu2_3d*mu2_3d - 315.*mu2_3d*mu2_3d + 105.*mu2_3d - 5.) )   ### ell = 6
    if ellmax > 5:
        L_ells = L_ells.at[4].set( 0.0078125 * (6435.*mu2_3d*mu2_3d*mu2_3d*mu2_3d  - 12012.*mu2_3d*mu2_3d*mu2_3d + 6930.*mu2_3d*mu2_3d  - 1260.*mu2_3d + 35.) ) ### ell = 8
    if ellmax > 8:
        L_ells = L_ells.at[5].set( 0.00390625* (46189.*mu2_3d*mu2_3d*mu2_3d*mu2_3d*mu2_3d - 109395.*mu2_3d*mu2_3d*mu2_3d*mu2_3d + 90090.*mu2_3d*mu2_3d*mu2_3d - 30030.*mu2_3d*mu2_3d + 3465.*mu2_3d - 63.) ) ### ell = 10
            
    ### compuute weighted power spectra
    Pks = jnp.zeros(nell*nbin)
    for ell in range(nell):
        legendre_fac = 2*2*ell+1
        Pk = pk_3d*legendre_fac*L_ells[ell]
        Pk = Pk.at[0,0,0].set(0.0)
                        
        k = k_3d.ravel()
        Nk = Nk_base.ravel()
        Pk = Pk.ravel()
            
        kidx = jnp.digitize(k, k_arr, right=True)
            
        Pk *= Nk
    
        Pk = jnp.bincount(kidx, weights=Pk, length=nbin+1)
        Nk = jnp.bincount(kidx, weights=Nk, length=nbin+1)
            
        Pk = Pk[1:nbin+1]
        Nk = Nk[1:nbin+1]
            
        Pk /= Nk
        
        Pks = Pks.at[ell*nbin:(ell+1)*nbin].set(Pk)

    return Pks
