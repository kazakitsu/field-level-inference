# !/usr/bin/env python3

import sys
import logging
import numpy as np

import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

import field_level.coord as coord
import field_level.assign_util as assign_util
import field_level.util as util
import field_level.cosmo_util as cosmo_util

class Forward_Model:
    def __init__(self, model_name, which_pk, ng_params, boxsize, space, **kwargs):
        self.model_name = model_name
        self.which_pk = which_pk
        if len(ng_params) == 3:
            self.ng, self.ng_L, self.ng_E = ng_params
        elif len(ng_params) == 4:
            self.ng, self.ng_L, self.ng_E, self.ng_max = ng_params
        self.ngo2 = self.ng // 2
        self.ng3_L = self.ng_L*self.ng_L*self.ng_L
        self.ng3_E = self.ng_E*self.ng_E*self.ng_E
        self.ngo2_E = self.ng_E // 2
        self._zeros_E = jnp.zeros((self.ng_E,)*3) # cached zero field
        self._norm_L  = float(self.ng_L**3)
        self._norm_E  = float(self.ng_E**3)

        self.boxsize = boxsize
        self.kf  = 2*jnp.pi/self.boxsize
        self.vol = self.boxsize**3
        self.space = space
        self.rsd_flag = False
        self.renormalize = True

        # batch FFT kernels (static vmap)
        self._irfftn_b = vmap(jnp.fft.irfftn, in_axes=0, out_axes=0)
        self._rfftn_b  = vmap(jnp.fft.rfftn,  in_axes=0, out_axes=0)

        print('model = ', self.model_name, file=sys.stderr)
        if   'gauss'     in model_name: self._branch = 0
        elif '1ept_G2'   in model_name: self._branch = 1
        elif '2ept'      in model_name: self._branch = 2
        elif 'lpt'       in model_name: self._branch = 3
        else:                           self._branch = 4   # fall‑back

        ### LPT pre-initial position
        pos_base_L = jnp.linspace(0, self.boxsize, self.ng_L, endpoint=False)
        self.pos_q_L = jnp.array(jnp.meshgrid(pos_base_L, pos_base_L, pos_base_L, indexing='ij'))

        if 'lpt' in model_name:
            self.window_order, self.interlace = kwargs['mas_params']
            if '1lpt' in model_name:
                self.lpt_order = 1
            elif '2lpt' in model_name:
                self.lpt_order = 2
        else:
            self.window_order = 1
            self.interlace = 0
            self.lpt_order = 0

        # assign() with fixed geometry/window is partially bound once
        self._assign_E0 = partial(
            assign_util.assign,
            self.boxsize,
            num_particles=self.ng3_L,
            window_order=self.window_order,
            interlace=0,           # fixed
            contd=0,
            max_scatter_indices=100_000_000
        )

        self._assign_E1 = partial(
            assign_util.assign,
            self.boxsize,
            num_particles=self.ng3_L,
            window_order=self.window_order,
            interlace=1,           # fixed
            contd=0,
            max_scatter_indices=100_000_000
        )

        if 'rsd' in model_name:
            print('rsd_flag = True', file=sys.stderr)
            self.rsd_flag = True

        if 'bare' in model_name:
            print('renormalize = False', file=sys.stderr)
            self.renormalize = False

    def kvecs(self, kmax):
        self.kvec = coord.rfftn_kvec([self.ng,]*3, self.boxsize, dtype=float)
        self.k2   = coord.rfftn_k2(self.kvec)

        self.kvec_E = coord.rfftn_kvec([self.ng_E,]*3, self.boxsize, dtype=float)
        self.k2_E   = coord.rfftn_k2(self.kvec_E)
        self.mu2_E = coord.rfftn_mu2(self.kvec_E)

        self.kvec_L  = coord.rfftn_kvec([self.ng_L,]*3, self.boxsize, dtype=float)
        self.kG1_L   = coord.rfftn_G1(self.kvec_L)
        self.kdisp_L = coord.rfftn_disp(self.kvec_L)
        
        nvec_E = self.kvec_E/self.kf
        self.Wk_E   = coord.deconvolve(nvec_E, self.window_order)    
        phase_E = jnp.pi*nvec_E.sum(axis=0)/self.ng_E
        self.phase_shift_E = jnp.cos(phase_E) + 1j*jnp.sin(phase_E)
        del nvec_E, phase_E
                        
        if kmax > 10.0:
            kmax = int(kmax)
            self.ng_max = kmax
            kvec_max = coord.rfftn_kvec([self.ng_max,]*3, self.boxsize, dtype=float)
            self.k2_max = coord.rfftn_k2(kvec_max)
            self.mu2_max = coord.rfftn_mu2(kvec_max)
            del kvec_max
        
    @partial(jit, static_argnames=('self',))
    def linear_power(self, cosmo_params):
        if 'cosmo' in self.which_pk:
            power_emu = CPJ(probe='mpk_lin')
            k_emu = power_emu.modes
            ### no massive neutrino
            Pk_emu = power_emu.predict(jnp.array(cosmo_params))
            hubble = cosmo_params[2]
            return [k_emu/hubble, Pk_emu*hubble**3]
        elif self.which_pk == 'pow': ###to do
            pass
        
    @partial(jit, static_argnames=('self', 'R', 'type_integ'))
    def sigmaR(self, pk_lin, R=8.0, type_integ='trap'):
        if type_integ == 'trap':
            x = pk_lin[0] * R
            wk = 3.*(jnp.sin(x)/x/x - jnp.cos(x)/x)/x
            sigma8 = jnp.sqrt(util.trapezoid(pk_lin[0]**3*pk_lin[1]*wk**2/(2.*jnp.pi**2),
                                             jnp.log(pk_lin[0])) )
        return sigma8.astype(float)
    
    @partial(jit, static_argnames=('self',))
    def linear_modes(self, cosmo_params, gauss_3d):
        if 'cosmo' in self.which_pk:
            k, pk = self.linear_power(cosmo_params)
            hubble = cosmo_params[2]
            pk_kvec = jnp.interp(jnp.sqrt(self.k2), 
                                 k, pk)
            pk_kvec = pk_kvec.at[0,0,0].set(0.0)
            return jnp.sqrt(pk_kvec/2.) * gauss_3d / jnp.sqrt(self.vol)
        elif self.which_pk == 'pow': ###to do
            pass
        
    @partial(jit, static_argnames=('self',))
    def _mass_to_grid(self, weight, pos_x, interlace):
        """
        Single call to assign(), interlace handled by flag (0/1).
        """
        def _case0(_):
            f = self._assign_E0(self._zeros_E, weight, pos_x)
            return jnp.fft.rfftn(f) / self._norm_E
        def _case1(_):
            f = self._assign_E1(self._zeros_E, weight, pos_x)
            return jnp.fft.rfftn(f) / self._norm_E

        return lax.cond(interlace == 1, _case1, _case0, operand=None)

    @partial(jit, static_argnames=('self',))
    def L2E(self, weight, pos_x):
        """Particle to Eulerian grid (k-space) with optional interlacing."""
        fieldk_0 = self._mass_to_grid(weight, pos_x, interlace=0)

        # add interlaced grid only if needed (static)
        def _with_interlace(_):
            fieldk_1 = self._mass_to_grid(weight, pos_x, interlace=1)
            fieldk_1 *= self.phase_shift_E
            return 0.5 * (fieldk_0 + fieldk_1)

        fieldk_E = lax.cond(self.interlace == 1,
                            _with_interlace,
                            lambda _: fieldk_0,
                            operand=None)

        return fieldk_E * self.Wk_E

    @partial(jit, static_argnames=('self',))
    def batch_irfftn(self, array_k):
        return self._irfftn_b(array_k) * self._norm_L

    @partial(jit, static_argnames=('self',))
    def batch_rfftn(self, array_r):
        return self._rfftn_b(array_r) / self._norm_L
    
    @partial(jit, static_argnames=('self',))
    def lpt(self, delk_L, growth_f=0.0):
        disp1k = self.kdisp_L*delk_L
        disp1r = self.batch_irfftn(disp1k)
        pos_x = self.pos_q_L + disp1r

        # optional RSD on 1LPT
        def _add_rsd1(p):  # p: pos_x
            return p.at[2].add(growth_f * disp1r[2])
        pos_x = lax.cond(self.rsd_flag, _add_rsd1, lambda p: p, pos_x)

        # 2LPT contribution if required
        if self.lpt_order == 2:
            phi2r = -0.5 * self.G2r(delk_L)
            disp2r = self.batch_irfftn(3./7. * self.kdisp_L * jnp.fft.rfftn(phi2r) / self._norm_L)
            pos_x  = pos_x + disp2r

            # RSD on 2LPT
            def _add_rsd2(p):
                return p.at[2].add(growth_f * disp2r[2])
            pos_x = lax.cond(self.rsd_flag, _add_rsd2, lambda p: p, pos_x)

        return pos_x

    def shift2r(self, delk_L):
        disp1k = self.kdisp_L*delk_L
        disp1r = self.batch_irfftn(disp1k)
        nablak = 1j * self.kvec_L * delk_L
        nablar = self.batch_irfftn(nablak)
        shift2r_L = jnp.sum(nablar * disp1r, axis=0)
        return - (shift2r_L - jnp.mean(shift2r_L))
    
    def d2r(self, delr_L):
        d2r_L = delr_L*delr_L
        return d2r_L - jnp.mean(d2r_L)
    
    def G2r(self, delk_L):
        G1k_L = self.kG1_L*delk_L
        G1r_L = self.batch_irfftn(G1k_L)
        phi2r_L = G1r_L[0]*G1r_L[3] + G1r_L[3]*G1r_L[5] + G1r_L[5]*G1r_L[0] - G1r_L[1]*G1r_L[1] - G1r_L[2]*G1r_L[2] - G1r_L[4]*G1r_L[4]
        ### 0:x, 1:y, 2:z
        ### 0: xx, 1: xy, 2: xz, 3: yy, 4: yz, 5: zz

        G2r_L   = -2.0*phi2r_L
        return G2r_L - jnp.mean(G2r_L)

    def d3r(self, delr_L, renormalize=True):
        d3r_L = delr_L*delr_L*delr_L
        ### the leading order renormalization
        #d2k_L = self.batch_rfftn(d2r_L)
        ### zero-padding
        #d2k_L = coord.func_extend(self.ng_L,
        #                          coord.func_reduce(self.ng, d2k_L))
        #d2r_L = self.batch_irfftn(d2k_L) 
        if renormalize:
            d2r_L = delr_L*delr_L
            sigma2 = jnp.mean(d2r_L)
            d3r_L = d3r_L - 3.0*sigma2*delr_L
        return d3r_L - jnp.mean(d3r_L)
    
    def G3r(self, delk_L, renormalize=True):
        G1k_L = self.kG1_L*delk_L
        G1r_L = self.batch_irfftn(G1k_L)
        ### 0:x, 1:y, 2:z
        ### 0: xx, 1: xy, 2: xz, 3: yy, 4: yz, 5: zz
        
        ### - Det(G1_ij)
        phi3ar_L = G1r_L[0]*G1r_L[4]*G1r_L[4] + G1r_L[3]*G1r_L[2]*G1r_L[2] + G1r_L[5]*G1r_L[1]*G1r_L[1] - 2.*G1r_L[1]*G1r_L[2]*G1r_L[4] - G1r_L[0]*G1r_L[3]*G1r_L[5]

        ### multiplying -1./3. results in one of the third order potential in LPT
        G3r_L   = 3.0*phi3ar_L
        return G3r_L - jnp.mean(G3r_L)
    
    def G2dr(self, delk_L, delr_L, renormalize=True):
        G2r_L = self.G2r(delk_L)
        #G2k_L = self.batch_rfftn(G2r_L)
        ### zero-padding
        #G2k_L = coord.func_extend(self.ng_L,
        #                          coord.func_reduce(self.ng, G2k_L))
        #G2r_L = self.batch_irfftn(G2k_L) 
        G2dr_L = delr_L * G2r_L

        ### the leading order renormalization
        if renormalize:
            d2r_L = delr_L*delr_L
            sigma2 = jnp.mean(d2r_L)
            G2dr_L = G2dr_L + 4./3.*sigma2*delr_L

        return G2dr_L - jnp.mean(G2dr_L)

    def Gamma3r(self, delk_L, delr_L, renormalize=True):
        G1k_L = self.kG1_L*delk_L
        G1r_L = self.batch_irfftn(G1k_L)

        phi2r_L = G1r_L[0]*G1r_L[3] + G1r_L[3]*G1r_L[5] + G1r_L[5]*G1r_L[0] - G1r_L[1]*G1r_L[1] - G1r_L[2]*G1r_L[2] - G1r_L[4]*G1r_L[4]
        G2r_L   = -2.0*phi2r_L
        G2r_L   = G2r_L - jnp.mean(G2r_L)

        G2k_L = jnp.fft.rfftn(G2r_L) / self.ng3_L
        G2k_L_ij = self.kG1_L * G2k_L

        G2r_L = self.batch_irfftn(G2k_L_ij)
        ### 0:x, 1:y, 2:z
        ### 0: xx, 1: xy, 2: xz, 3: yy, 4: yz, 5: zz
        ### 00: 0, 01: 1, 02: 2, 11: 3, 12: 4, 22: 5

        phi3br_L = 0.5*G1r_L[0]*(G2r_L[3]+G2r_L[5]) + 0.5*G1r_L[3]*(G2r_L[5]+G2r_L[0]) + 0.5*G1r_L[5]*(G2r_L[0]+G2r_L[3]) - G1r_L[1]*G2r_L[1] - G1r_L[2]*G2r_L[2] - G1r_L[4]*G2r_L[4]
        ### multiplying -10./21. results in one of the third order potential in LPT
        ### Gamma3 = -8/7 \phi^(3b)
        #Gamma3r_L = -8./7.*phi3br_L
        Gamma3r_L = -8./7.*phi3br_L

        ### the leading order renormalization

        #if renormalize:
        #    d2r_L = delr_L*delr_L
        #    sigma2 = d2r_L.mean()
        #    Gamma3r_L = Gamma3r_L + 32./35.*sigma2*delr_L
        return Gamma3r_L
    
    def full_cubic_r(self, delk_L, delr_L):
        d2r_L = delr_L*delr_L
        sigma2 = jnp.mean(d2r_L)
        d2r_L = d2r_L - sigma2

        G1k_L = self.kG1_L*delk_L
        G1r_L = self.batch_irfftn(G1k_L)

        phi2r_L = G1r_L[0]*G1r_L[3] + G1r_L[3]*G1r_L[5] + G1r_L[5]*G1r_L[0] - G1r_L[1]*G1r_L[1] - G1r_L[2]*G1r_L[2] - G1r_L[4]*G1r_L[4]
        G2r_L   = -2.0*phi2r_L
        G2r_L   = G2r_L - jnp.mean(G2r_L)

        # d3
        d3r_L = delr_L*delr_L*delr_L
        # G2d
        G2dr_L = delr_L * G2r_L

        # G3
        ### - Det(G1_ij)
        phi3ar_L = G1r_L[0]*G1r_L[4]*G1r_L[4] + G1r_L[3]*G1r_L[2]*G1r_L[2] + G1r_L[5]*G1r_L[1]*G1r_L[1] - 2.*G1r_L[1]*G1r_L[2]*G1r_L[4] - G1r_L[0]*G1r_L[3]*G1r_L[5]
        G3r_L   = 3.0*phi3ar_L

        # Gamma3
        G2k_L = jnp.fft.rfftn(G2r_L) / self.ng3_L
        G2k_L_ij = self.kG1_L * G2k_L

        G2r_L = self.batch_irfftn(G2k_L_ij)
        ### 0:x, 1:y, 2:z
        ### 0: xx, 1: xy, 2: xz, 3: yy, 4: yz, 5: zz
        ### 00: 0, 01: 1, 02: 2, 11: 3, 12: 4, 22: 5

        phi3br_L = 0.5*G1r_L[0]*(G2r_L[3]+G2r_L[5]) + 0.5*G1r_L[3]*(G2r_L[5]+G2r_L[0]) + 0.5*G1r_L[5]*(G2r_L[0]+G2r_L[3]) - G1r_L[1]*G2r_L[1] - G1r_L[2]*G2r_L[2] - G1r_L[4]*G2r_L[4]
        ### multiplying -10./21. results in one of the third order potential in LPT
        ### Gamma3 = -8/7 \phi^(3b)
        Gamma3r_L = -8./7.*phi3br_L

        if self.renormalize:
            d3r_L = d3r_L - 3.0*sigma2*delr_L
            G2dr_L = G2dr_L + 4./3.*sigma2*delr_L
        
        return d2r_L, G2r_L, d3r_L - jnp.mean(d3r_L), G2dr_L - jnp.mean(G2dr_L), G3r_L - jnp.mean(G3r_L), Gamma3r_L - jnp.mean(Gamma3r_L)

    # -----------------------------
    # Helper functions to further modularize compute_model
    # -----------------------------
    def _compute_gauss_model(self, delk_L, biases):
        """
        Compute the field for a 'gauss' model.
        """
        if 'lin' in self.model_name:
            b1 = biases[0]
        else:
            b1 = 1.0
        if 'rsd' in self.model_name:
            growth_f = biases[-1]
            kaiser_fac = b1 + growth_f * self.mu2_E
        else:
            kaiser_fac = b1
        if self.ng_E < self.ng_L:
            fieldk_E = kaiser_fac * coord.func_reduce(self.ng_E, delk_L)
        else:
            fieldk_E = kaiser_fac * coord.func_extend(self.ng_E, delk_L)
        return fieldk_E

    def _compute_1ept_G2_model(self, delk_L, biases):
        """
        Compute the field for a '1ept_G2' model.
        """
        G2r_L = self.G2r(delk_L)
        G2k_L = jnp.fft.rfftn(G2r_L) / self.ng3_L
        delk_E = coord.func_reduce(self.ng_E, delk_L)
        G2k_E = coord.func_reduce(self.ng_E, G2k_L)
        if 'lin' in self.model_name:
            b1 = biases[0]
        else:
            b1 = 1.0
        if 'rsd' in self.model_name:
            growth_f = biases[-1]
            kaiser_fac = b1 + growth_f * self.mu2_E
        else:
            kaiser_fac = b1
        fieldk_E = kaiser_fac * (delk_E + G2k_E)
        return fieldk_E

    def _compute_2ept_model(self, delk_L, biases):
        """
        Compute the field for a '2ept' model.
        """
        delr_L = jnp.fft.irfftn(delk_L) * self.ng3_L
        d2k_L = jnp.fft.rfftn(self.d2r(delr_L)) / self.ng3_L

        shift2r_L = self.shift2r(delk_L) ### This is - \Psi * \nabla \delta
        shift2k_L = jnp.fft.rfftn(shift2r_L) / self.ng3_L

        G2r_L = self.G2r(delk_L)
        G2k_L = jnp.fft.rfftn(G2r_L) / self.ng3_L
        
        delk_E = coord.func_reduce(self.ng_E, delk_L)
        d2k_E = coord.func_reduce(self.ng_E, d2k_L)
        shift2k_E = coord.func_reduce(self.ng_E, shift2k_L)
        G2k_E = coord.func_reduce(self.ng_E, G2k_L)
        
        if 'lin' in self.model_name:
            b1 = biases[0]
            if 'c0' in self.model_name:
                c0 = biases[1]
                fieldk_E = (b1 + c0 * self.k2_E) * delk_E + b1 * (d2k_E + shift2k_E + 2./7. * G2k_E)
            else:
                fieldk_E = b1 * (delk_E + d2k_E + shift2k_E + 2./7. * G2k_E)
        elif 'quad' in self.model_name:
            b1 = biases[0]
            b2 = biases[1]
            bG2 = biases[2]
            if 'c0' in self.model_name:
                c0 = biases[3]
                fieldk_E = (b1 + c0 * self.k2_E) * delk_E + b1 * shift2k_E + (b1 + 0.5 * b2) * d2k_E + (2./7. * b1 + bG2) * G2k_E
            else:
                fieldk_E = b1 * (delk_E + shift2k_E) + (b1 + 0.5 * b2) * d2k_E + (2.0/7.0 * b1 + bG2) * G2k_E
        else:
            fieldk_E = delk_E + d2k_E + shift2k_E + 2.0/7.0 * G2k_E
        return fieldk_E

    def _compute_lpt_model(self, delk_L, biases):
        """
        Compute the field for an LPT model.
        """
        delr_L = jnp.fft.irfftn(delk_L) * self.ng3_L

        if self.rsd_flag:
            pos_x = self.lpt(delk_L, growth_f=biases[-1])
        else:
            pos_x = self.lpt(delk_L)

        if 'matter' in self.model_name:
            weight = jnp.ones((self.ng_L, self.ng_L, self.ng_L))
        else:
            weight = jnp.zeros((self.ng_L, self.ng_L, self.ng_L))

        if 'lin' in self.model_name:
            b1 = biases[0]
            if 'c0' in self.model_name:
                c0 = biases[1]
                factor_lin = b1 + c0 * self.k2_E
                if 'c2' in self.model_name:
                    c2 = biases[2]
                    factor_lin += c2 * self.k2_E * self.mu2_E
                    if 'c4' in self.model_name:
                        c4 = biases[3]
                        factor_lin += c4 * self.k2_E * self.mu2_E * self.mu2_E
                fieldk_E = self.L2E(weight, pos_x) + factor_lin * self.L2E(delr_L, pos_x)
            else:
                weight += b1 * delr_L
                fieldk_E = self.L2E(weight, pos_x)
        elif 'quad' in self.model_name:
            b1 = biases[0]
            factor_lin = b1
            b2 = biases[1]
            bG2 = biases[2]
            d2r_L = self.d2r(delr_L)
            G2r_L = self.G2r(delk_L)
            weight += 0.5 * b2 * d2r_L + bG2 * G2r_L
            if 'c0' in self.model_name:
                c0 = biases[3]
                factor_lin = b1 + c0 * self.k2_E
                if 'c2' in self.model_name:
                    c2 = biases[4]
                    factor_lin += c2 * self.k2_E * self.mu2_E
                    if 'c4' in self.model_name:
                        c4 = biases[5]
                        factor_lin += c4 * self.k2_E * self.mu2_E * self.mu2_E
                fieldk_E = self.L2E(weight, pos_x) + factor_lin * self.L2E(delr_L, pos_x)
            else:
                weight += b1 * delr_L
                fieldk_E = self.L2E(weight, pos_x)
        elif 'cubic' in self.model_name:
            b1 = biases[0]
            factor_lin = b1
            b2 = biases[1]
            bG2 = biases[2]
            b3 = biases[3]
            bG2d = biases[4]
            bG3 = biases[5]
            bGamma3 = biases[6]
            d2r_L, G2r_L, d3r_L, G2dr_L, G3r_L, Gamma3r_L = self.full_cubic_r(delk_L, delr_L)
            weight += 0.5 * b2 * d2r_L + bG2 * G2r_L + b3 * d3r_L + bG2d * G2dr_L + bG3 * G3r_L + bGamma3 * Gamma3r_L
            if 'c0' in self.model_name:
                c0 = biases[7]
                factor_lin = b1 + c0 * self.k2_E
                if 'c2' in self.model_name:
                    c2 = biases[8]
                    factor_lin += c2 * self.k2_E * self.mu2_E
                    if 'c4' in self.model_name:
                        c4 = biases[9]
                        factor_lin += c4 * self.k2_E * self.mu2_E * self.mu2_E
                fieldk_E = self.L2E(weight, pos_x) + factor_lin * self.L2E(delr_L, pos_x)
            else:
                weight += b1 * delr_L
                fieldk_E = self.L2E(weight, pos_x)
        else:  ### matter
            factor_lin = 0.0
            if 'c0' in self.model_name:
                c0 = biases[0]
                factor_lin += c0 * self.k2_E
                if 'c2' in self.model_name:
                    c2 = biases[1]
                    factor_lin += c2 * self.k2_E * self.mu2_E
                    if 'c4' in self.model_name:
                        c4 = biases[2]
                        factor_lin += c4 * self.k2_E * self.mu2_E * self.mu2_E
            fieldk_E = factor_lin * self.L2E(weight, pos_x)
        return fieldk_E

    @partial(jit, static_argnames=('self',))
    def compute_model(self, delk_L, biases, *vals):
            
        def _gauss(_):
            return self._compute_gauss_model(delk_L, biases)

        def _1ept(_):
            return self._compute_1ept_G2_model(delk_L, biases)

        def _2ept(_):
            return self._compute_2ept_model(delk_L, biases)

        def _lpt(_):
            return self._compute_lpt_model(delk_L, biases)

        # fallback
        def _dummy(_):
            return jnp.zeros_like(coord.func_reduce(self.ng_E, delk_L))

        fieldk_E = lax.switch(
            self._branch,                     # static
            (_gauss, _1ept, _2ept, _lpt, _dummy),
            operand=None                      
        )

        if self.space == 'k_space':
            fieldk_E = fieldk_E.at[0,0,0].set(0.0)
            return fieldk_E
        elif self.space == 'r_space':
            fieldk_E = fieldk_E.at[0,0,0].set(0.0)
            ng3_max = self.ng_max ** 3
            fieldk_max = coord.func_reduce(self.ng_max, fieldk_E)
            fieldr_max = jnp.fft.irfftn(fieldk_max) * ng3_max
            delk_max = coord.func_reduce(self.ng_max, delk_L)
            delr_max = jnp.fft.irfftn(delk_max) * ng3_max
            d2r_max = delr_max * delr_max
            return fieldr_max, delr_max, d2r_max
