#!/usr/bin/env python3

import jax.numpy as jnp
import jax

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

    bmax = jnp.digitize(kmax, k_arr, right=True)
    k  = k[1:bmax+1]
    Pk = Pk[1:bmax+1]
    Nk = Nk[1:bmax+1]
    k_arr = k_arr[:bmax+1]
    
    k /= Nk
    Pk /= Nk
    
    Pk *= vol

    return k, Pk, Nk

def window_legendre_compute(fieldk, boxsize, nbin=60, kmin=0.0, kmax=0.6, ellmax = 6):
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
            
            bmax = jnp.digitize(kmax, k_arr, right=True)
            k  = k[1:bmax+1]
            Pk = Pk[1:bmax+1]
            Nk = Nk[1:bmax+1]
            
            k /= Nk
            Pk /= Nk
    
            window = window.at[ell1, ell2].set(Pk)
            
    return window
