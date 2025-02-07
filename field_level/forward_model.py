# !/usr/bin/env python3

import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from field_level.JAX_Zenbu import Zenbu
from field_level.Zenbu_utils.loginterp_jax import loginterp_jax

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
        elif len(ng_params) == 5:
            self.ng, self.ng_L, self.ng_E, self.ng_cut, self.ng_e = ng_params
        self.ngo2 = int(self.ng/2)
        self.ng3_L = self.ng_L*self.ng_L*self.ng_L
        self.ng3_E = self.ng_E*self.ng_E*self.ng_E
        self.ngo2_E = int(self.ng_E/2)
        self.boxsize = boxsize
        self.kf  = 2*jnp.pi/self.boxsize
        self.vol = self.boxsize**3
        self.space = space
        self.initialized_Zenbu = False
        self.initialized_lin_pk = False
        self.rsd_flag = False
        print('model = ', self.model_name, file=sys.stderr)
        if 'lpt' in model_name:
            self.window_order, self.interlace = kwargs['mas_params']
            ### LPT pre-initial position
            pos_base_L = jnp.linspace(0, self.boxsize, self.ng_L, endpoint=False)
            self.pos_q_L = jnp.array(jnp.meshgrid(pos_base_L, pos_base_L, pos_base_L, indexing='ij'))
            if '1lpt' in model_name:
                self.lpt_order = 1
            elif '2lpt' in model_name:
                self.lpt_order = 2
        if 'rsd' in model_name:
            self.rsd_flag = True

    #@partial(jit, static_argnums=(0,))
    def kvecs(self, kmax):
        self.kvec = coord.rfftn_kvec([self.ng, self.ng, self.ng], self.boxsize, dtype=float)
        self.k2   = coord.rfftn_k2(self.kvec)

        self.kvec_E = coord.rfftn_kvec([self.ng_E, self.ng_E, self.ng_E], self.boxsize, dtype=float)
        self.k2_E   = coord.rfftn_k2(self.kvec_E)
        self.mu2_E = self.kvec_E[2]*self.kvec_E[2]/self.k2_E
        self.mu2_E = self.mu2_E.at[0, 0, 0].set(0.0)

        if 'ept' in self.model_name:
            self.kvec_L  = coord.rfftn_kvec([self.ng_L, self.ng_L, self.ng_L], self.boxsize, dtype=float)
            self.kG1_L   = coord.rfftn_G1(self.kvec_L)
            self.kdisp_L = coord.rfftn_disp(self.kvec_L)
        
        if 'lpt' in self.model_name:
            kvec_L  = coord.rfftn_kvec([self.ng_L, self.ng_L, self.ng_L], self.boxsize, dtype=float)
            self.kG1_L   = coord.rfftn_G1(kvec_L)
            #k2_L    = coord.rfftn_k2(kvec_L)
            self.kdisp_L = coord.rfftn_disp(kvec_L)
        
            nvec_E = self.kvec_E/self.kf
            self.Wk_E   = coord.deconvolve(nvec_E, self.window_order)    
            phase_E = jnp.pi*nvec_E.sum(axis=0)/self.ng_E
            self.phase_shift_E = jnp.cos(phase_E) + 1j*jnp.sin(phase_E)
            del nvec_E, phase_E
                        
        if kmax > 10.0:
            kmax = int(kmax)
            self.ng_max = kmax
            kvec_max = coord.rfftn_kvec([self.ng_max, self.ng_max, self.ng_max], self.boxsize, dtype=float)
            self.k2_max = coord.rfftn_k2(kvec_max)
            self.mu2_max = kvec_max[2]*kvec_max[2]/self.k2_max
            self.mu2_max = self.mu2_max.at[0, 0, 0].set(0.0)
        
    #@partial(jit, static_argnums=(0,))
    def linear_power(self, cosmo_params):
        if not self.initialized_lin_pk:
            if 'cosmo' in self.which_pk:
                self.power_emu = CPJ(probe='mpk_lin')
                self.k = self.power_emu.modes
                ### no massive neutrino
                Pk_emu = self.power_emu.predict(jnp.array(cosmo_params))
                hubble = cosmo_params[2]
                return jnp.array([self.k/hubble, Pk_emu*hubble**3])
            elif self.which_pk == 'pow': ###to do
                pass
            self.initialized_lin_pk = True
        else:
            if 'cosmo' in self.which_pk:
                Pk_emu = self.power_emu.predict(jnp.array(cosmo_params))
                hubble = cosmo_params[2]
                return jnp.array([self.k/hubble, Pk_emu*hubble**3])
        
    @partial(jit, static_argnums=(0, 2))
    def sigma8(self, pk_lin, type_integ='trap'):
        def sigma8_integ(ln_k):
            k = jnp.exp(ln_k)
            pk = loginterp_jax(pk_lin[0], pk_lin[1])(k)
            x = k*8.0
            wk = 3.*(jnp.sin(x)/x/x - jnp.cos(x)/x)/x
            return k**3*pk*wk**2/(2.*jnp.pi**2)
        if type_integ == 'trap':
            x = 8.0*pk_lin[0]
            wk = 3.*(jnp.sin(x)/x/x - jnp.cos(x)/x)/x
            sigma8 = jnp.sqrt(util.trapezoid(pk_lin[0]**3*pk_lin[1]*wk**2/(2.*jnp.pi**2),
                                             jnp.log(pk_lin[0])) )
        elif type_integ == 'simps':  ### not working
            sigma8 = jnp.sqrt(util.simps(sigma8_integ,
                                         jnp.log(pk_lin[0]),
                                         jnp.log(pk_lin[-1]), ))
        elif type_integ == 'romb':  ### not working
            sigma8 = jnp.sqrt(util.romb(sigma8_integ,
                                        jnp.log(pk_lin[0]),
                                        jnp.log(pk_lin[-1]), ))
        return sigma8.astype(float)
    
    @partial(jit, static_argnums=(0,))
    def linear_modes(self, cosmo_params, gauss_3d):
        if 'cosmo' in self.which_pk:
            pk = self.power_emu.predict(jnp.array(cosmo_params))
            hubble = cosmo_params[2]
            pk_kvec = jnp.interp(jnp.sqrt(self.k2), 
                                 self.k/hubble, pk*hubble*hubble*hubble)
            pk_kvec = pk_kvec.at[0,0,0].set(0.0)
            #gauss_1d_re, gauss_1d_im = coord.gauss_to_delta(gauss_1d, self.ng)
            #gauss_3d = gauss_1d_re.reshape(self.ng, self.ng, self.ngo2+1) + 1j*gauss_1d_im.reshape(self.ng, self.ng, self.ngo2+1)
            return jnp.sqrt(pk_kvec/2.) * gauss_3d / jnp.sqrt(self.vol)
        elif self.which_pk == 'pow': ###to do
            pass

    def call_Zenbu(self, cosmo_params, kmax):
        print('Preparing the function to compute transfer functions...', file=sys.stderr)
        self.extrap_min = -3.5
        self.extrap_max = 1.5
        self.N = 500
        
        hubble = cosmo_params[2]
        self.kint = jnp.logspace(self.extrap_min, self.extrap_max, self.N)
        #print('self.k.shape = ', self.k.shape, file=sys.stderr)
        #print('self.power_emu.predict(cosmo_params).shape = ', self.power_emu.predict(cosmo_params).shape, file=sys.stderr)
        pint_base = loginterp_jax(self.k/hubble,
                                  self.power_emu.predict(jnp.array(cosmo_params))*hubble*hubble*hubble)(self.kint)
        
        kk_min = np.sqrt(self.k2_E[0,0,1]).min()*1.0
        if kmax > 10.0:
            kk_max = jnp.pi/self.boxsize*kmax*jnp.sqrt(3.0)
        else:
            kk_max = kmax*1.0
        
        NN = 30 
        self.modPT = Zenbu(self.kint, pint_base, kmin=kk_min, kmax=kk_max, nk=NN, jn=5)
        print('Done.', file=sys.stderr)
    
    @partial(jit, static_argnums=(0, ))
    def transfer_function(self, pk_lin):
    #def transfer_function(self, cosmo_params, kmax, pk_lin):
        ### This pk_lin should (k [Mpc/h], P(k) [(h/Mpc)^3]) unit
        '''
        if not self.initialized_Zenbu:
            print('Preparing the transfer functions...', file=sys.stderr)
            extrap_min = -3.5
            extrap_max = 1.5
            N = 500
        
            hubble = cosmo_params[2]
            #print('self.k.shape = ', self.k.shape, file=sys.stderr)
            #print('self.power_emu.predict(cosmo_params).shape = ', self.power_emu.predict(jnp.array(cosmo_params)).shape, file=sys.stderr)
            kint = jnp.logspace(extrap_min, extrap_max, N)
            pint_base = loginterp_jax(self.k/hubble,
                                      self.power_emu.predict(jnp.array(cosmo_params))*hubble*hubble*hubble)(kint)
        
            kk_min = self.kf*0.9
            if kmax > 10.0:
                kk_max = jnp.pi/self.boxsize*kmax*jnp.sqrt(3.0)
            else:
                kk_max = kmax*1.0
        
            NN = 30 
            self.modPT = Zenbu(kint, pint_base, kmin=kk_min, kmax=kk_max, nk=NN, jn=5)
            self.initialized_Zenbu = True
            print('Done.', file=sys.stderr)
        else:
        '''
        pint = loginterp_jax(pk_lin[0], pk_lin[1])(self.kint)
        self.modPT.update_power_spectrum(pint)
        ptable = self.modPT.make_ptable()
        pdd_E = jnp.interp(np.sqrt(self.k2_E), ptable[:,0], ptable[:,1])
        p1Gamma3_E = -2.*jnp.interp(np.sqrt(self.k2_E), ptable[:,0], ptable[:,2])/pdd_E
        p1S3_E = jnp.interp(np.sqrt(self.k2_E), ptable[:,0], ptable[:,3])/pdd_E
        return pdd_E, p1Gamma3_E, p1S3_E
    
    @partial(jit, static_argnums=(0,))
    def L2E(self, weight, pos_x):
        fieldr_E = jnp.zeros((self.ng_E, self.ng_E, self.ng_E))
        fieldr_E = assign_util.assign(self.boxsize, fieldr_E, weight, pos_x, self.window_order, interlace=0)
        fieldk_E = jnp.fft.rfftn(fieldr_E)/self.ng3_E

        if self.interlace == 1:
            fieldr_E_ = jnp.zeros((self.ng_E, self.ng_E, self.ng_E))
            fieldr_E_ = assign_util.assign(self.boxsize, fieldr_E_, weight, pos_x, self.window_order, interlace=1)
            fieldk_E_ = jnp.fft.rfftn(fieldr_E_)/self.ng3_E
            fieldk_E_ *= self.phase_shift_E
            fieldk_E = (fieldk_E + fieldk_E_)*0.5
        
        fieldk_E *= self.Wk_E
        return fieldk_E

    @partial(jit, static_argnums=(0,))
    def compute_3irfft(self, array_k):
        ng_ = array_k.shape[1]
        array_r = jnp.zeros((3, ng_, ng_, ng_))

        def body_fun(i, array_r):
            return array_r.at[i].set(jnp.fft.irfftn(array_k[i]))

        array_r = lax.fori_loop(0, 3, body_fun, array_r)
        return array_r

    @partial(jit, static_argnums=(0,))
    def compute_6irfft(self, array_k):
        ng_ = array_k.shape[1]
        array_r = jnp.zeros((6, ng_, ng_, ng_))

        def body_fun(i, array_r):
            return array_r.at[i].set(jnp.fft.irfftn(array_k[i]))

        array_r = lax.fori_loop(0, 6, body_fun, array_r)
        return array_r
    
    @partial(jit, static_argnums=(0,))
    def lpt(self, delk_L, growth_f=0.0):
        disp1k = self.kdisp_L*delk_L
        disp1r = self.compute_3irfft(disp1k)
        #disp1r = jnp.array([jnp.fft.irfftn(disp1k[0]),
        #                    jnp.fft.irfftn(disp1k[1]),
        #                    jnp.fft.irfftn(disp1k[2])])
        disp1r *=  self.ng3_L
        pos_x = self.pos_q_L + disp1r
        if self.lpt_order == 2:
            phi2r_L = -0.5*self.G2r(delk_L)
            phi2k_L = jnp.fft.rfftn(phi2r_L) / self.ng3_L
            disp2k_L = 3./7. * self.kdisp_L * phi2k_L
            disp2r_L = self.compute_3irfft(disp2k_L)
            #disp2r_L = jnp.array([jnp.fft.irfftn(disp2k_L[0]),
            #                      jnp.fft.irfftn(disp2k_L[1]),
            #                      jnp.fft.irfftn(disp2k_L[2])])
            disp2r_L *= self.ng3_L
            pos_x = pos_x + disp2r_L
        if self.rsd_flag:  ### only the zeldovich is supported in this rsd part 
            pos_x = pos_x.at[2].add( growth_f * disp1r[2] )
        return pos_x

    @partial(jit, static_argnums=(0,))
    def shift2r(self, delk_L):
        disp1k = self.kdisp_L*delk_L
        disp1r = self.compute_3irfft(disp1k)
        #disp1r = jnp.array([jnp.fft.irfftn(disp1k[0]),
        #                    jnp.fft.irfftn(disp1k[1]),
        #                    jnp.fft.irfftn(disp1k[2])])
        disp1r *=  self.ng3_L
        nablak = 1j * self.kvec_L * delk_L
        nablar = self.compute_3irfft(nablak)
        #nablar = jnp.array([jnp.fft.irfftn(nablak[0]),
        #                    jnp.fft.irfftn(nablak[1]),
        #                    jnp.fft.irfftn(nablak[2])])
        nablar *=  self.ng3_L
        shift2r_L = jnp.sum(nablar * disp1r, axis=0)
        shift2r_L -= shift2r_L.mean()
        return -shift2r_L
    
    @partial(jit, static_argnums=(0,))
    def d2r(self, delr_L):
        d2r_L = delr_L*delr_L
        #d2r_L = d2r_L - d2r_L.mean()
        d2r_L -= d2r_L.mean()
        return d2r_L
    
    @partial(jit, static_argnums=(0,))
    def G2r(self, delk_L):
        G1k_L = self.kG1_L*delk_L
        G1r_L = self.compute_6irfft(G1k_L) * self.ng3_L
        phi2r_L = G1r_L[0]*G1r_L[3] + G1r_L[3]*G1r_L[5] + G1r_L[5]*G1r_L[0] - G1r_L[1]*G1r_L[1] - G1r_L[2]*G1r_L[2] - G1r_L[4]*G1r_L[4]
        ### 0:x, 1:y, 2:z
        #G1r00_L = jnp.fft.irfftn(G1k_L[0]) * self.ng3_L
        #G1r01_L = jnp.fft.irfftn(G1k_L[1]) * self.ng3_L
        #G1r02_L = jnp.fft.irfftn(G1k_L[2]) * self.ng3_L
        #G1r11_L = jnp.fft.irfftn(G1k_L[3]) * self.ng3_L
        #G1r12_L = jnp.fft.irfftn(G1k_L[4]) * self.ng3_L
        #G1r22_L = jnp.fft.irfftn(G1k_L[5]) * self.ng3_L
        
        #phi2r_L = G1r00_L*G1r11_L + G1r11_L*G1r22_L + G1r22_L*G1r00_L - G1r01_L*G1r01_L - G1r02_L*G1r02_L - G1r12_L*G1r12_L

        G2r_L   = -2.0*phi2r_L
        G2r_L   -= G2r_L.mean()
        #G2r_L   = G2r_L - G2r_L.mean()
        return G2r_L

    @partial(jit, static_argnums=(0,))
    def d3r(self, delr_L):
        d3r_L = delr_L*delr_L*delr_L
        ### the leading order renormalization
        #d2r_L = delr_L*delr_L
        #sigma = d2r_L.mean()
        #d3r_L = d3r_L - 3.0*sigma*delr_L
        return d3r_L

    @partial(jit, static_argnums=(0,))
    def G3r(self, delk_L):
        G1k_L = self.kG1_L*delk_L
        G1r00_L = jnp.fft.irfftn(G1k_L[0]) * self.ng3_L
        G1r01_L = jnp.fft.irfftn(G1k_L[1]) * self.ng3_L
        G1r02_L = jnp.fft.irfftn(G1k_L[2]) * self.ng3_L
        G1r11_L = jnp.fft.irfftn(G1k_L[3]) * self.ng3_L
        G1r12_L = jnp.fft.irfftn(G1k_L[4]) * self.ng3_L
        G1r22_L = jnp.fft.irfftn(G1k_L[5]) * self.ng3_L

        ### - Det(G1_ij)
        phi3ar_L = G1r00_L*G1r12_L*G1r12_L + G1r11_L*G1r02_L*G1r02_L + G1r22_L*G1r01_L*G1r01_L - 2.*G1r01_L*G1r02_L*G1r12_L - G1r00_L*G1r11_L*G1r22_L
        ### multiplying -1./3. results in one of the third order potential in LPT
        G3r_L   = 3.0*phi3ar_L
        G3r_L   -= G3r_L.mean()
        return G3r_L
    
    @partial(jit, static_argnums=(0,))
    def G2dr(self, delk_L, delr_L):
        G2r_L = self.G2r(delk_L)
        G2dr_L = delr_L * G2r_L
        ### the leading order renormalization
        #d2r_L = delr_L*delr_L
        #sigma = d2r_L.mean()
        #G2dr_L -= 4./3.*sigma*delr_L

        G2dr_L -= G2dr_L.mean()
        return G2dr_L

    @partial(jit, static_argnums=(0,))
    def Gamma3r(self, delk_L):
        G1k_L = self.kG1_L*delk_L
        G1r00_L = jnp.fft.irfftn(G1k_L[0]) * self.ng3_L
        G1r01_L = jnp.fft.irfftn(G1k_L[1]) * self.ng3_L
        G1r02_L = jnp.fft.irfftn(G1k_L[2]) * self.ng3_L
        G1r11_L = jnp.fft.irfftn(G1k_L[3]) * self.ng3_L
        G1r12_L = jnp.fft.irfftn(G1k_L[4]) * self.ng3_L
        G1r22_L = jnp.fft.irfftn(G1k_L[5]) * self.ng3_L

        phi2r_L = G1r00_L*G1r11_L + G1r11_L*G1r22_L + G1r22_L*G1r00_L - G1r01_L*G1r01_L - G1r02_L*G1r02_L - G1r12_L*G1r12_L
        G2r_L   = -2.0*phi2r_L
        G2r_L   -= G2r_L.mean()
        G2k_L = jnp.fft.rfftn(G2r_L) / self.ng3_L
        G2k_L_ij = self.kG1_L * G2k_L

        G2r00_L = jnp.fft.irfftn(G2k_L_ij[0]) * self.ng3_L
        G2r01_L = jnp.fft.irfftn(G2k_L_ij[1]) * self.ng3_L
        G2r02_L = jnp.fft.irfftn(G2k_L_ij[2]) * self.ng3_L
        G2r11_L = jnp.fft.irfftn(G2k_L_ij[3]) * self.ng3_L
        G2r12_L = jnp.fft.irfftn(G2k_L_ij[4]) * self.ng3_L
        G2r22_L = jnp.fft.irfftn(G2k_L_ij[5]) * self.ng3_L

        phi3br_L = 0.5*G1r00_L*(G2r11_L+G2r22_L) + 0.5*G1r11_L*(G2r22_L+G2r00_L) + 0.5*G1r22_L*(G2r00_L+G2r11_L) - G1r01_L*G2r01_L - G1r12_L*G2r12_L - G1r02_L*G2r02_L 
        ### multiplying -10./21. results in one of the third order potential in LPT
        ### Gamma3 = -4/7 \phi^(3b)
        Gamma3r_L   = -4./7.*phi3br_L

        ### the leading order renormalization
        #d2r_L = delr_L*delr_L
        #sigma = d2r_L.mean()
        #Gamma3r_L = Gamma3r_L + 32./35.*sigma*delr_L
        return Gamma3r_L
    
    ### under construction
    @partial(jit, static_argnums=(0,))
    def full_cubic_op(self, delk_L, delr_L):
        G1k_L = self.kG1_L*delk_L
        G1r00_L = jnp.fft.irfftn(G1k_L[0]) * self.ng3_L
        G1r01_L = jnp.fft.irfftn(G1k_L[1]) * self.ng3_L
        G1r02_L = jnp.fft.irfftn(G1k_L[2]) * self.ng3_L
        G1r11_L = jnp.fft.irfftn(G1k_L[3]) * self.ng3_L
        G1r12_L = jnp.fft.irfftn(G1k_L[4]) * self.ng3_L
        G1r22_L = jnp.fft.irfftn(G1k_L[5]) * self.ng3_L

        phi2r_L = G1r00_L*G1r11_L + G1r11_L*G1r22_L + G1r22_L*G1r00_L - G1r01_L*G1r01_L - G1r02_L*G1r02_L - G1r12_L*G1r12_L
        G2r_L   = -2.0*phi2r_L
        G2r_L   -= G2r_L.mean()
        G2k_L = jnp.fft.rfftn(G2r_L) / self.ng3_L
        G2k_L_ij = self.kG1_L * G2k_L

        G2r00_L = jnp.fft.irfftn(G2k_L_ij[0]) * self.ng3_L
        G2r01_L = jnp.fft.irfftn(G2k_L_ij[1]) * self.ng3_L
        G2r02_L = jnp.fft.irfftn(G2k_L_ij[2]) * self.ng3_L
        G2r11_L = jnp.fft.irfftn(G2k_L_ij[3]) * self.ng3_L
        G2r12_L = jnp.fft.irfftn(G2k_L_ij[4]) * self.ng3_L
        G2r22_L = jnp.fft.irfftn(G2k_L_ij[5]) * self.ng3_L

        phi3br_L = 0.5*G1r00_L*(G2r11_L+G2r22_L) + 0.5*G1r11_L*(G2r22_L+G2r00_L) + 0.5*G1r22_L*(G2r00_L+G2r11_L) - G1r01_L*G2r01_L - G1r12_L*G2r12_L - G1r02_L*G2r02_L 
        ### multiplying -10./21. results in one of the third order potential in LPT
        ### Gamma3 = -4/7 \phi^(3b)
        Gamma3r_L   = -4./7.*phi3br_L

        ### the leading order renormalization
        #d2r_L = delr_L*delr_L
        #sigma = d2r_L.mean()
        #Gamma3r_L = Gamma3r_L + 32./35.*sigma*delr_L
        return Gamma3r_L
        
    @partial(jit, static_argnums=(0, ))
    def models(self, delk_L, biases, *vals):
        if  'gauss' in self.model_name:
            if 'lin' in self.model_name:
                b1 = biases[0]
                if 'rsd' in self.model_name:
                    growth_f = biases[-1]
                    kaiser_fac = b1 + growth_f *self.mu2_E
                else:
                    kaiser_fac = b1
            else:
                kaiser_fac = 1.0
            if self.ng_E < self.ng_L:
                fieldk_E = kaiser_fac * coord.reduce_deltak(self.ng_E, delk_L)
            else:
                fieldk_E = kaiser_fac * coord.func_extend(self.ng_E, delk_L)
        elif '1ept_G2' in self.model_name:
            G2r_L = self.G2r(delk_L)
            G2k_L = jnp.fft.rfftn(G2r_L)/self.ng3_L
            delk_E = coord.reduce_deltak(self.ng_E, delk_L)
            G2k_E = coord.reduce_deltak(self.ng_E, G2k_L)
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
        elif '2ept' in self.model_name:
            delr_L = jnp.fft.irfftn(delk_L)*self.ng3_L
            d2r_L = self.d2r(delr_L)
            d2k_L = jnp.fft.rfftn(d2r_L)/self.ng3_L
            shift2r_L = self.shift2r(delk_L) ### This is - \Psi * \nabla \delta
            shift2k_L = jnp.fft.rfftn(shift2r_L)/self.ng3_L
            G2r_L = self.G2r(delk_L)
            G2k_L = jnp.fft.rfftn(G2r_L)/self.ng3_L
            delk_E = coord.reduce_deltak(self.ng_E, delk_L)
            d2k_E = coord.reduce_deltak(self.ng_E, d2k_L)
            shift2k_E = coord.reduce_deltak(self.ng_E, shift2k_L)
            G2k_E = coord.reduce_deltak(self.ng_E, G2k_L)
            if 'lin' in self.model_name:
                b1 = biases[0]
                if 'cs2' in self.model_name:
                    cs2 = biases[1]
                    fieldk_E = (b1 + cs2*self.k2_E) * delk_E + b1 * (d2k_E + shift2k_E + 2./7.*G2k_E)
                else:
                    fieldk_E = b1 * (delk_E + d2k_E + shift2k_E + 2./7.*G2k_E)
            elif 'quad' in self.model_name:
                b1 = biases[0]
                b2 = biases[1]
                bG2 = biases[2]
                if 'cs2' in self.model_name:
                    cs2 = biases[3]
                    fieldk_E = (b1 + cs2*self.k2_E) * delk_E + b1 * shift2k_E + (b1 + 0.5*b2) * d2k_E + (2./7.*b1 + bG2) * G2k_E
                else:
                    fieldk_E = b1 * (delk_E + shift2k_E) + (b1 + 0.5*b2) * d2k_E + (2./7.*b1 + bG2) * G2k_E
            else:
                fieldk_E = delk_E + d2k_E + shift2k_E + 2./7.*G2k_E
        elif '1lpt' in self.model_name:
            delr_L = jnp.fft.irfftn(delk_L)*self.ng3_L
            if self.rsd_flag:
                pos_x = self.lpt(delk_L, growth_f=biases[-1])
            else:
                pos_x = self.lpt(delk_L)
            if 'matter' in self.model_name:
                weight_zel = jnp.ones((self.ng_L, self.ng_L, self.ng_L))
                fieldk_zel = self.L2E(weight_zel, pos_x)
            else:
                #fieldk_E = jnp.zeros((self.ng_E, self.ng_E, self.ngo2_E+1))
                fieldk_zel = jnp.zeros((self.ng_E, self.ng_E, self.ngo2_E+1))
            #if 'c1' in self.model_name or 'c2' in self.model_name :
            #    pos_x_ = self.lpt(delk_L, growth_f=0.0)
            #    weight_zel_ = jnp.ones((self.ng_L, self.ng_L, self.ng_L))
            #    fieldk_zel_ = self.L2E(weight_zel_, pos_x_)
            if 'lin' in self.model_name:
                b1 = biases[0]
                if 'cs2' in self.model_name and 'c1' in self.model_name and 'c2' in self.model_name:
                    cs2 = biases[1]
                    c1 = biases[2]
                    c2 = biases[3]
                    fieldk_E = fieldk_zel + ( b1 + cs2*self.k2_E + c1*self.k2_E*self.mu2_E + c2*self.k2_E*self.mu2_E*self.mu2_E ) * self.L2E(delr_L, pos_x)
                    #fieldk_E = fieldk_zel + b1 * self.L2E(delr_L, pos_x) + ( cs2*self.k2_E + c1*self.k2_E*self.mu2_E + c2*self.k2_E*self.mu2_E*self.mu2_E ) * fieldk_zel_
                elif 'cs2' in self.model_name and 'c1' in self.model_name:
                    cs2 = biases[1]
                    c1 = biases[2]
                    fieldk_E = fieldk_zel + ( b1 + cs2*self.k2_E + c1*self.k2_E*self.mu2_E ) * self.L2E(delr_L, pos_x)
                    #fieldk_E = fieldk_zel + b1 * self.L2E(delr_L, pos_x) + ( cs2*self.k2_E + c1*self.k2_E*self.mu2_E ) * fieldk_zel_
                elif 'cs2' in self.model_name:
                    cs2 = biases[1]
                    fieldk_E = fieldk_zel + ( b1 + cs2*self.k2_E ) * self.L2E(delr_L, pos_x)
                    #fieldk_E = fieldk_zel + b1 * self.L2E(delr_L, pos_x) + cs2*self.k2_E * fieldk_zel
                else:
                    fieldk_E = fieldk_zel + b1 * self.L2E(delr_L, pos_x)
            elif 'quad' in self.model_name:
                b1 = biases[0]
                b2 = biases[1]
                bG2 = biases[2]
                d2r_L = self.d2r(delr_L)
                G2r_L = self.G2r(delk_L)
                weight_lin = delr_L
                weight_quad = b2 * d2r_L / 2. +  bG2 * G2r_L
                fieldk_lin  = self.L2E(weight_lin,  pos_x)
                fieldk_quad = self.L2E(weight_quad, pos_x)
                if 'cs2' in self.model_name and 'c1' in self.model_name and 'c2' in self.model_name:
                    cs2 = biases[3]
                    c1 = biases[4]
                    c2 = biases[5]
                    fieldk_E = fieldk_zel + ( b1 + cs2*self.k2_E + c1*self.k2_E*self.mu2_E + c2*self.k2_E*self.mu2_E*self.mu2_E ) * fieldk_lin + fieldk_quad
                    #fieldk_E = fieldk_zel + fieldk_lin + fieldk_quad + ( cs2*self.k2_E + c1*self.k2_E*self.mu2_E + c2*self.k2_E*self.mu2_E*self.mu2_E ) * fieldk_zel_
                elif 'cs2' in self.model_name and 'c1' in self.model_name:
                    cs2 = biases[3]
                    c1 = biases[4]
                    fieldk_E = fieldk_zel + ( b1 + cs2*self.k2_E + c1*self.k2_E*self.mu2_E ) * fieldk_lin + fieldk_quad
                    #fieldk_E = fieldk_zel + fieldk_lin + fieldk_quad + ( cs2*self.k2_E + c1*self.k2_E*self.mu2_E ) * fieldk_zel_
                elif 'cs2' in self.model_name:
                    cs2 = biases[3]
                    fieldk_E = fieldk_zel + ( b1 + cs2*self.k2_E ) * fieldk_lin + fieldk_quad
                    #fieldk_E = fieldk_zel + b1 * fieldk_lin + fieldk_quad + cs2*self.k2_E * fieldk_zel_
                else:
                    fieldk_E = fieldk_zel + fieldk_lin + fieldk_quad
            elif 'cubic' in self.model_name:
                b1 = biases[0]
                b2 = biases[1]
                bG2 = biases[2]
                bGamma3 = biases[3]
                pk_lin = vals[0]
                pddk_E, p1Gamma3k_E, p1S3k_E = self.transfer_function(pk_lin)
                p1Gamma3k_E = p1Gamma3k_E/pddk_E
                p1S3k_E = p1S3k_E/pddk_E
                d2r_L = self.d2r(delr_L)
                G2r_L = self.G2r(delk_L)
                if 'cs2' in self.model_name and 'c1' in self.model_name and 'c2' in self.model_name:
                    cs2 = biases[4]
                    c1 = biases[5]
                    c2 = biases[6]
                    fieldk_E = fieldk_E + ( b1 + cs2*self.k2_E + c1*self.k2_E*self.mu2_E + c1*self.k2_E*self.mu2_E*self.mu2_E + bGamma3*p1Gamma3k_E + b1*p1S3k_E ) * self.L2E(delr_L, pos_x) + self.L2E(0.5*b2*d2r_L +  bG2*G2r_L, pos_x)
                elif 'cs2' in self.model_name and 'c1' in self.model_name:
                    cs2 = biases[4]
                    c1 = biases[5]
                    fieldk_E = fieldk_E + ( b1 + cs2*self.k2_E + c1*self.k2_E*self.mu2_E + bGamma3*p1Gamma3k_E + b1*p1S3k_E ) * self.L2E(delr_L, pos_x) + self.L2E(0.5*b2*d2r_L +  bG2*G2r_L, pos_x)
                elif 'cs2' in self.model_name:
                    cs2 = biases[4]
                    fieldk_E = fieldk_E + ( b1 + cs2*self.k2_E + bGamma3*p1Gamma3k_E + b1*p1S3k_E ) * self.L2E(delr_L, pos_x) + self.L2E(0.5*b2*d2r_L +  bG2*G2r_L, pos_x)
                else:
                    fieldk_E = fieldk_E + ( b1 + bGamma3*p1Gamma3k_E + b1*p1S3k_E ) * self.L2E(delr_L, pos_x) + self.L2E(0.5*b2*d2r_L +  bG2*G2r_L, pos_x)
            else: ### pure matter
                if 'cs2' in self.model_name and 'c1' in self.model_name and 'c2' in self.model_name:
                    cs2 = biases[0]
                    c1 = biases[1]
                    c2 = biases[2]
                    fieldk_E = ( 1. + cs2*self.k2_E + c1*self.k2_E*self.mu2_E + c2*self.k2_E*self.mu2_E*self.mu2_E ) * fieldk_zel
                elif 'cs2' in self.model_name and 'c1' in self.model_name:
                    cs2 = biases[0]
                    c1 = biases[1]
                    fieldk_E = ( 1. + cs2*self.k2_E + c1*self.k2_E*self.mu2_E ) * fieldk_zel
                elif 'cs2' in self.model_name:
                    cs2 = biases[0]
                    fieldk_E = ( 1. + cs2*self.k2_E ) * fieldk_zel
                else:
                    fieldk_E = fieldk_zel
        if self.space == 'k_space':
            fieldk_E = fieldk_E.at[0,0,0].set(0.0)
            return fieldk_E
        elif self.space == 'r_space':
            fieldk_E = fieldk_E.at[0,0,0].set(0.0)
            ng3_max = self.ng_max ** 3
            fieldk_max = coord.reduce_deltak(self.ng_max, fieldk_E)
            fieldr_max = jnp.fft.irfftn(fieldk_max) * ng3_max
            delk_max = coord.reduce_deltak(self.ng_max, delk_L)
            delr_max = jnp.fft.irfftn(delk_max) * ng3_max
            d2r_max = delr_max * delr_max
            return fieldr_max, delr_max, d2r_max
