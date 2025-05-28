#!/usr/bin/env python3

import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

import field_level.coord as coord

class Measure_Pk:
    def __init__(self, boxsize, ng, kbin_1d, ell_max=0, leg_fac=True):
        self.boxsize = boxsize
        self.kbin_1d = jnp.array(kbin_1d)
        #self.kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])
        self.num_bins = self.kbin_1d.shape[0] - 1
        self.vol     = self.boxsize**3
        self.ng      = ng
        ### precompute k-vector grid
        kvec         = coord.rfftn_kvec([self.ng,]*3, self.boxsize)
        k2           = (kvec ** 2).sum(axis=0)
        kmag         = jnp.sqrt(k2)
        self.kmag_1d   = kmag.ravel()
        ### per-mode segment indices
        self.kidx = jnp.digitize(self.kmag_1d, self.kbin_1d, right=True)
        ### per-mode counts
        Nk = jnp.full_like(k2, 2, dtype=jnp.int32)
        Nk = Nk.at[..., 0].set(1)
        if k2.shape[-1] % 2 == 0:
            Nk = Nk.at[..., -1].set(1)
        self.Nk_1d = Nk.ravel()

        self.k_mean = jnp.bincount(self.kidx, weights=self.kmag_1d * self.Nk_1d, length=self.num_bins+1)[1:]
        self.Nk = jnp.bincount(self.kidx, weights=self.Nk_1d, length=self.num_bins+1)[1:]
        self.k_mean /= self.Nk

        ### mu^2 for multipoles
        mu2 = (kvec[2]**2) / k2
        mu2 = mu2.at[0,0,0].set(0.0)
        self.mu2_1d = mu2.ravel()
        self.leg_fac = leg_fac
        legs = []
        for ell in range(0, ell_max+1, 2):
            if self.leg_fac:
                legendre_fac = 2.0 * ell + 1.0
            else:
                legendre_fac = 1.0
            if ell == 0:
                legs.append(legendre_fac * jnp.ones_like(self.mu2_1d))
            elif ell == 2:
                legs.append(legendre_fac * 0.5 * (3.0*self.mu2_1d - 1.0))
            elif ell == 4:
                legs.append(legendre_fac * 0.125 * (35.0*self.mu2_1d*self.mu2_1d - 30.0*self.mu2_1d + 3.0))
        self.legendre_stack = jnp.stack(legs, axis=0)

    def _compute(self, fieldk1, fieldk2, ell):
        Pk_1d = (fieldk1 * fieldk2.conj()) * self.Nk_1d
        Pk_1d = Pk_1d.at[0].set(0.0)

        idx = ell // 2
        legendre_weight = self.legendre_stack[idx]

        Pk_1d = Pk_1d * legendre_weight

        Pk = jnp.bincount(self.kidx, weights=Pk_1d, length=self.num_bins+1)[1:]

        Pk /= self.Nk
        Pk *= self.vol

        return self.k_mean, Pk, self.Nk

    @partial(jit, static_argnames=('self','ell'))
    def pk_auto(self, fieldk, ell=0):
        return self._compute(fieldk.ravel(), fieldk.ravel(), ell)

    @partial(jit, static_argnames=('self','ell'))
    def pk_cross(self, fieldk1, fieldk2, ell=0):
        return self._compute(fieldk1.ravel(), fieldk2.ravel(), ell)


def power_compute(fieldk_1, fieldk_2, boxsize, nbin=60, kmin=0.0, kmax=0.6, ell=0, leg_fac=True):
    nbin = int(nbin)
    ng = fieldk_1.shape[0]
    if fieldk_1.shape != fieldk_2.shape:
        raise ValueError(f'{fieldk_1.shape} != {fieldk_2.shape}')

    kf = 2*jnp.pi/boxsize
    kvec = coord.rfftn_kvec([ng,]*3, boxsize)
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

    #print('Nk = ', Nk)

    k  = k[1:nbin+1]
    Pk = Pk[1:nbin+1]
    Nk = Nk[1:nbin+1]
    k_arr = k_arr[:nbin+1]
    
    k /= Nk
    Pk /= Nk
    
    Pk *= vol

    return k, Pk, Nk

class Measure_spectra_FFT:
    def __init__(self, boxsize, ng, kbin_1d, bispec=True, open_triangle=False):
        self.boxsize = boxsize
        self.kbin_1d = jnp.array(kbin_1d)
        self.vol     = self.boxsize**3
        self.ng      = ng
        kvec         = coord.rfftn_kvec([ng,]*3, self.boxsize)
        self.kmag    = jnp.sqrt(coord.rfftn_k2(kvec))
        self.kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])
        self.num_bins = self.kbin_1d.shape[0] - 1

        if bispec:
            triangle_idx_list = []

            kmin_bins = self.kbin_1d[:-1]
            kmax_bins = self.kbin_1d[1:]

            for i in range(self.num_bins):
                for j in range(i, self.num_bins):
                    for k in range(j, self.num_bins):
                        k1 = float(self.kbin_centers[i])
                        k2 = float(self.kbin_centers[j])
                        k3 = float(self.kbin_centers[k])

                        k1min = float(kmin_bins[i])
                        k1max = float(kmax_bins[i])
                        k2min = float(kmin_bins[j])
                        k2max = float(kmax_bins[j])
                        k3min = float(kmin_bins[k])
                        k3max = float(kmax_bins[k])
                        
                        if open_triangle:
                            if ((k1max + k2max > k3min) and 
                                (k2max + k3max > k1min) and 
                                (k3max + k1max > k2min)):
                                triangle_idx_list.append((i, j, k))
                        else:
                            if k3 >= abs(k1 - k2) and k3 <= (k1 + k2):
                                triangle_idx_list.append((i, j, k))
            self.triangle_idxs = jnp.array(triangle_idx_list)
            self.k123 = jnp.array([self.kbin_centers[self.triangle_idxs[:,0]], 
                                   self.kbin_centers[self.triangle_idxs[:,1]], 
                                   self.kbin_centers[self.triangle_idxs[:,2]]]).T

    def filter_field(self, fieldk, kmin, kmax):
        mask = (kmin <= self.kmag) & (self.kmag < kmax)
        fieldk_filtered = fieldk * mask
        fieldr = jnp.fft.irfftn(fieldk_filtered) * (self.ng**3)
        return fieldr

    def measure_pk_bk(self, fieldk,):
        num_bins = len(self.kbin_1d) - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])

        onesk = jnp.ones_like(fieldk)

        def filter_all_fields(kmin, kmax):
            fieldr = self.filter_field(fieldk, kmin, kmax)
            onesr   = self.filter_field(onesk,  kmin, kmax)
            return fieldr, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        Ix_fields, norm_fields = vmap(filter_all_fields)(kmins, kmaxs)

        def compute_triangle(idx):
            i, j, k = idx
            k1 = kbin_centers[i]
            k2 = kbin_centers[j]
            k3 = kbin_centers[k]

            valid = jnp.logical_and(k3 >= jnp.abs(k1 - k2), k3 <= (k1 + k2))

            def true_fn(_):
                product_field = Ix_fields[i] * Ix_fields[j] * Ix_fields[k]
                bispec = jnp.sum(product_field) * self.vol * self.vol
                normalization = norm_fields[i] * norm_fields[j] * norm_fields[k]
                num_triangles = jnp.sum(normalization)
                bispec /= num_triangles
                return jnp.array([k1, k2, k3, bispec.real, num_triangles / self.ng**3])
            return lax.cond(valid, true_fn, lambda _: jnp.zeros(5), operand=None)

        triangles = vmap(compute_triangle)(self.triangle_idxs)

        def compute_power(i):
            product_field = Ix_fields[i] * Ix_fields[i]
            power = jnp.sum(product_field) * self.vol
            normalization = norm_fields[i] ** 2
            num_modes = jnp.sum(normalization)
            power /= num_modes
            return jnp.array([kbin_centers[i], power.real, num_modes / self.ng**3])
        
        lines = vmap(compute_power)(jnp.arange(num_bins))

        return lines, triangles

    @partial(jit, static_argnames=('self', 'batch_size'))
    def measure_bk(self, fieldk, batch_size=20):
        num_bins = len(self.kbin_1d) - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])

        onesk = jnp.ones_like(fieldk)

        def filter_all_fields(kmin, kmax):
            fieldr = self.filter_field(fieldk, kmin, kmax)
            onesr  = self.filter_field(onesk,  kmin, kmax)
            return fieldr, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        Ix_fields, norm_fields = vmap(filter_all_fields)(kmins, kmaxs)

        def compute_triangle(idx):
            i, j, k = idx
            k1 = kbin_centers[i]
            k2 = kbin_centers[j]
            k3 = kbin_centers[k]

            valid = jnp.logical_and(k3 >= jnp.abs(k1 - k2), k3 <= (k1 + k2))

            def true_fn(_):
                product_field = Ix_fields[i] * Ix_fields[j] * Ix_fields[k]
                bispec = jnp.sum(product_field) * self.vol * self.vol
                normalization = norm_fields[i] * norm_fields[j] * norm_fields[k]
                num_triangles = jnp.sum(normalization)
                bispec /= num_triangles
                return jnp.array([k1, k2, k3, bispec.real, num_triangles / self.ng**3])
            return lax.cond(valid, true_fn, lambda _: jnp.zeros(5), operand=None)

        n_triangles = self.triangle_idxs.shape[0]
        triangles_batches = []
        for start in range(0, n_triangles, batch_size):
            end = start + batch_size
            batch_idxs = self.triangle_idxs[start:end]
            batch_triangles = vmap(compute_triangle)(batch_idxs)
            triangles_batches.append(batch_triangles)
        triangles = jnp.concatenate(triangles_batches, axis=0)

        return triangles

    @partial(jit, static_argnames=('self'))
    def measure_pk(self, fieldk):
        num_bins = self.kbin_1d.shape[0] - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])
        onesk = jnp.ones_like(fieldk)

        def filter_two_fields(kmin, kmax):
            fieldr = self.filter_field(fieldk, kmin, kmax)
            onesr  = self.filter_field(onesk,  kmin, kmax)
            return fieldr, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        Ix_fields, norm_fields = vmap(filter_two_fields)(kmins, kmaxs)

        def compute_power(i):
            product_field = Ix_fields[i] * Ix_fields[i]
            power = jnp.sum(product_field) * self.vol
            normalization = norm_fields[i] ** 2
            num_modes = jnp.sum(normalization)
            power /= num_modes
            return jnp.array([kbin_centers[i], power.real, num_modes / self.ng**3])
        
        lines = vmap(compute_power)(jnp.arange(num_bins))
        return lines


    @partial(jit, static_argnames=('self',))
    def measure_bispectrum_(self, field1k, field2k, field3k):
        num_bins = len(self.kbin_1d) - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])

        onesk = jnp.ones_like(field1k)

        def filter_all_fields(kmin, kmax):
            field1r = self.filter_field(field1k, kmin, kmax)
            field2r = self.filter_field(field2k, kmin, kmax)
            field3r = self.filter_field(field3k, kmin, kmax)
            onesr   = self.filter_field(onesk,  kmin, kmax)
            return field1r, field2r, field3r, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        I1x_fields, I2x_fields, I3x_fields, norm_fields = vmap(filter_all_fields)(kmins, kmaxs)

        def compute_triangle(idx):
            i, j, k = idx
            k1 = kbin_centers[i]
            k2 = kbin_centers[j]
            k3 = kbin_centers[k]

            valid = jnp.logical_and(k3 >= jnp.abs(k1 - k2), k3 <= (k1 + k2))

            def true_fn(_):
                product_field = I1x_fields[i] * I2x_fields[j] * I3x_fields[k]
                bispec = jnp.sum(product_field) * self.vol * self.vol
                normalization = norm_fields[i] * norm_fields[j] * norm_fields[k]
                num_triangles = jnp.sum(normalization)
                bispec /= num_triangles
                return jnp.array([k1, k2, k3, bispec.real, num_triangles / self.ng**3])
            return lax.cond(valid, true_fn, lambda _: jnp.zeros(5), operand=None)

        triangles = vmap(compute_triangle)(self.triangle_idxs)

        return triangles


    @partial(jit, static_argnames=('self',))
    def measure_power_(self, field1k, field2k):
        num_bins = self.kbin_1d.shape[0] - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])
        onesk = jnp.ones_like(field1k)

        def filter_two_fields(kmin, kmax):
            field1r = self.filter_field(field1k, kmin, kmax)
            field2r = self.filter_field(field2k, kmin, kmax)
            onesr   = self.filter_field(onesk,  kmin, kmax)
            return field1r, field2r, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        I1x_fields, I2x_fields, norm_fields = vmap(filter_two_fields)(kmins, kmaxs)

        def compute_power(i):
            product_field = I1x_fields[i] * I2x_fields[i]
            power = jnp.sum(product_field) * self.vol
            normalization = norm_fields[i] ** 2
            num_modes = jnp.sum(normalization)
            power /= num_modes
            return jnp.array([kbin_centers[i], power.real, num_modes / self.ng**3])
        
        lines = vmap(compute_power)(jnp.arange(num_bins))
        return lines


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
