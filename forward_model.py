# !/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
import sys
import gc
import jax
import time
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
from JAX_Zenbu import Zenbu
from Zenbu_utils.loginterp_jax import loginterp_jax

jax.config.update("jax_enable_x64", True)

import coord
import assign_util
import util
import cosmo_util


class Forward_Model:
    def __init__(self, model_name, which_pk, ng_params, boxsize, space, **kwargs):
        self.model_name = model_name
        self.which_pk = which_pk
        if len(ng_params) == 3:
            self.ng, self.ng_L, self.ng_E = ng_params
        elif len(ng_params) == 5:
            self.ng, self.ng_L, self.ng_E, self.ng_cut, self.ng_e = ng_params
        self.ngo2 = int(self.ng/2)
        self.ng3_E = self.ng_E*self.ng_E*self.ng_E
        self.boxsize = boxsize
        self.kf  = 2*jnp.pi/self.boxsize
        self.vol = self.boxsize**3
        self.space = space
        self.initialized_Zenbu = False
        self.initialized_lin_pk = False
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
            if 'zelrsd' in model_name:
                self.rsd_flag = True

    #@partial(jit, static_argnums=(0,))
    def kvecs(self, kmax):
        self.kvec = coord.rfftn_kvec([self.ng, self.ng, self.ng], self.boxsize, dtype=float)
        self.k2   = coord.rfftn_k2(self.kvec)

        self.kvec_E = coord.rfftn_kvec([self.ng_E, self.ng_E, self.ng_E], self.boxsize, dtype=float)
        self.k2_E   = coord.rfftn_k2(self.kvec_E)
        
        if 'lpt' in self.model_name:
            kvec_L  = coord.rfftn_kvec([self.ng_L, self.ng_L, self.ng_L], self.boxsize, dtype=float)
            #k2_L    = coord.rfftn_k2(kvec_L)
            self.kdisp_L = coord.rfftn_disp(kvec_L)
            self.kG2_L   = coord.rfftn_G2(kvec_L)
        
            nvec_E = self.kvec_E/self.kf
            self.Wk_E   = coord.deconvolve(nvec_E, self.window_order)    
            phase_E = jnp.pi*nvec_E.sum(axis=0)/self.ng_E
            self.phase_shift_E = jnp.cos(phase_E) + 1j*jnp.sin(phase_E)
            del nvec_E, phase_E
            
        if 'rsd' in self.model_name:
            self.mu2_E = self.kvec_E[2]*self.kvec_E[2]/self.k2_E
            self.mu2_E = self.mu2_E.at[0, 0, 0].set(0.0)
            
        if kmax > 10.0:
            kmax = int(kmax)
            self.ng_max = kmax
            kvec_max = coord.rfftn_kvec([self.ng_max, self.ng_max, self.ng_max], self.boxsize, dtype=float)
            self.k2_max = coord.rfftn_k2(kvec_max)
            self.mu2_max = self.kvec_max[2]*kvec_max[2]/self.k2_max
            self.mu2_max = self.mu2_max.at[0, 0, 0].set(0.0)
        
    #@partial(jit, static_argnums=(0,))
    def linear_power(self, cosmo_params):
        if not self.initialized_lin_pk:
            if 'cosmo' in self.which_pk:
                self.power_emu = CPJ(probe='mpk_lin')
                self.k = self.power_emu.modes
                ### no massive neutrino
                Pk_emu = self.power_emu.predict(jnp.array(cosmo_params))
                return jnp.array([self.k, Pk_emu])
            elif self.which_pk == 'pow': ###to do
                pass
            self.initialized_lin_pk = True
        else:
            Pk_emu = self.power_emu.predict(jnp.array(cosmo_params))
            return jnp.array([self.k, Pk_emu])
        
    @partial(jit, static_argnums=(0, 3))
    def sigma8(self, cosmo_params, pk_lin, type_integ='simps'):
        omega_b, omega_c, hubble = cosmo_params[:3]
        OM = (omega_b + omega_c)/hubble/hubble
        redshift = cosmo_params[-1]
        #if redshift != 0.0:
        #    raise ValueError('z != 0.0 in the sigma8 computation!')
        def sigma8_integ(ln_k):
            k = jnp.exp(ln_k)
            pk = loginterp_jax(pk_lin[0]/hubble, pk_lin[1]*hubble**3)(k)
            x = k*8.0
            wk = 3.*(jnp.sin(x)/x/x - jnp.cos(x)/x)/x
            return k**3*pk*wk**2/(2.*jnp.pi**2)
        if type_integ == 'trap':
            x = 8.0*pk_lin[0]/hubble
            wk = 3.*(jnp.sin(x)/x/x - jnp.cos(x)/x)/x
            sigma8 = jnp.sqrt(util.trapezoid(pk_lin[0]**3*pk_lin[1]*wk**2/(2.*jnp.pi**2),
                                             jnp.log(pk_lin[0]/hubble)) )
            ### this tmp_sigma8 is @ z=redshift, not @ z=0.0
            #sigma8 /= cosmo_util.growth_D(redshift, OM)
        elif type_integ == 'simps':
            sigma8 = jnp.sqrt(util.simps(sigma8_integ,
                                         jnp.log(pk_lin[0]/hubble),
                                         jnp.log(pk_lin[-1]/hubble), ))
            ### this tmp_sigma8 is @ z=redshift, not @ z=0.0
            #sigma8 /= cosmo_util.growth_D(redshift, OM)
        elif type_integ == 'romb':
            sigma8 = jnp.sqrt(util.romb(sigma8_integ,
                                        jnp.log(pk_lin[0]/hubble),
                                        jnp.log(pk_lin[-1]/hubble), ))
            ### this tmp_sigma8 is @ z=redshift, not @ z=0.0
            #sigma8 /= cosmo_util.growth_D(redshift, OM)
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
        print('Preparing the transfer functions...', file=sys.stderr)
        extrap_min = -3.5
        extrap_max = 1.5
        N = 500
        
        hubble = cosmo_params[2]
        kint = jnp.logspace(extrap_min, extrap_max, N)
        #print('self.k.shape = ', self.k.shape, file=sys.stderr)
        #print('self.power_emu.predict(cosmo_params).shape = ', self.power_emu.predict(cosmo_params).shape, file=sys.stderr)
        pint_base = loginterp_jax(self.k/hubble,
                                  self.power_emu.predict(jnp.array(cosmo_params))*hubble*hubble*hubble)(kint)
        
        kk_min = np.sqrt(self.k2_E[0,0,1]).min()*1.0
        if kmax > 10.0:
            kk_max = jnp.pi/self.boxsize*kmax*jnp.sqrt(3.0)
        else:
            kk_max = kmax*1.0
        
        NN = 30 
        self.modPT = Zenbu(kint, pint_base, kmin=kk_min, kmax=kk_max, nk=NN, jn=5)
    
    @partial(jit, static_argnums=(0, ))
    def transfer_function(self, cosmo_params, pk_lin):
    #def transfer_function(self, cosmo_params, kmax, pk_lin):
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
        hubble =  cosmo_params[2]
        self.modPT.update_power_spectrum(pk_lin[1]*hubble**3)
        ptable = self.modPT.make_ptable()
        pdd_E = jnp.interp(np.sqrt(self.k2_E)/hubble, ptable[:,0], ptable[:,1])
        p1Gamma3_E = -2.*jnp.interp(np.sqrt(self.k2_E)/hubble, ptable[:,0], ptable[:,2])/pdd_E
        p1S3_E = jnp.interp(np.sqrt(self.k2_E)/hubble, ptable[:,0], ptable[:,3])/pdd_E
        return pdd_E, p1Gamma3_E, p1S3_E
    
    @partial(jit, static_argnums=(0,))
    def L2E(self, weight, pos_x):
        fieldr_E = jnp.zeros((self.ng_E, self.ng_E, self.ng_E))
        fieldr_E = assign_util.assign(self.boxsize, self.ng_E, weight, pos_x, fieldr_E, self.window_order, 0)
        fieldk_E = jnp.fft.rfftn(fieldr_E)/self.ng3_E

        if self.interlace == 1:
            fieldr_E_ = jnp.zeros((self.ng_E, self.ng_E, self.ng_E))
            fieldr_E_ = assign_util.assign(self.boxsize, self.ng_E, weight, pos_x, fieldr_E_, self.window_order, 0)
            fieldk_E_ = jnp.fft.rfftn(fieldr_E_)/self.ng3_E
            fieldk_E_ *= self.phase_shift_E
            fieldk_E = (fieldk_E + fieldk_E_)*0.5
        
        fieldk_E *= self.Wk_E
        return fieldk_E
    
    
    @partial(jit, static_argnums=(0,))
    def lpt(self, delk_L, *vals):
        disp1k = self.kdisp_L*delk_L
        disp1r = jnp.array([jnp.fft.irfftn(disp1k[0]),
                            jnp.fft.irfftn(disp1k[1]),
                            jnp.fft.irfftn(disp1k[2])])
        disp1r *=  self.ng3_L
        pos_x = self.pos_q_L + disp1r
        if self.lpt_order == 2:
            phi2r_L = -0.5*self.G2r(delk_L)
            phi2k_L = jnp.fft.rfftn(phi2r_L) / self.ng3_L
            disp2k_L = 3./7. * self.kdisp_L * phi2k_L
            disp2r_L = jnp.array([jnp.fft.irfftn(disp2k_L[0]),
                                  jnp.fft.irfftn(disp2k_L[1]),
                                  jnp.fft.irfftn(disp2k_L[2])])
            disp2r_L *= self.ng3_L
            pos_x += disp2r_L
        if self.rsd_flag == True:  ### only the zeldovich is supported in this rsd part 
            growth_f = vals[0]
            pos_x = pos_x.at[2].add(growth_f*disp1r[2])
        return pos_x
    
    @partial(jit, static_argnums=(0,))
    def d2r(self, delr_L):
        d2r_L = delr_L*delr_L
        d2r_L -= d2r_L.mean()
        return d2r_L
    
    @partial(jit, static_argnums=(0,))
    def G2r(self, delk_L):
        G1k_L = self.kG2_L*delk_L
        G1r00_L = jnp.fft.irfftn(G1k_L[0]) * self.ng3_L
        G1r10_L = jnp.fft.irfftn(G1k_L[1]) * self.ng3_L
        G1r20_L = jnp.fft.irfftn(G1k_L[2]) * self.ng3_L
        G1r11_L = jnp.fft.irfftn(G1k_L[3]) * self.ng3_L
        G1r21_L = jnp.fft.irfftn(G1k_L[4]) * self.ng3_L
        G1r22_L = jnp.fft.irfftn(G1k_L[5]) * self.ng3_L
        
        phi2r_L = G1r00_L*G1r11_L + G1r11_L*G1r22_L + G1r22_L*G1r00_L - G1r10_L*G1r10_L - G1r20_L*G1r20_L - G1r21_L*G1r21_L
        G2r_L   = -2.0*phi2r_L
        G2r_L -= G2r_L.mean()
        return G2r_L
    
    @partial(jit, static_argnums=(0,))
    def models(self, delk_L, biases, *vals):
        if self.model_name == 'gauss':
            if self.ng_E < self.ng_L:
                fieldk_E = coord.reduce_deltak(self.ng_E, delk_L)
            else:
                fieldk_E = coord.func_extend(self.ng_E, delk_L)
        elif self.model_name == 'gauss_rsd':
            b1 = biases[0]
            growth_f = vals[0]
            kaiser_fac = b1 + growth_f * self.mu2_E
            if self.ng_E < self.ng_L:
                fieldk_E = coord.reduce_deltak(self.ng_E, delk_L)
            else:
                fieldk_E = coord.func_extend(self.ng_E, delk_L)
            fieldk_E *= kaiser_fac
        elif '1lpt' in self.model_name:
            delr_L = jnp.fft.irfftn(delk_L)*self.ng3_L
            if self.rsd_flag:
                pos_x = self.lpt(self, delk_L, vals)
            else:
                pos_x = self.lpt(self, delk_L)
            if 'lin' in self.model_name:
                b1 = biases[0]
                if 'cs2' in self.model_name:
                    cs2 = biases[1]
                    fieldk_E = ( b1 + cs2*self.k2_E ) * self.L2E(self, delr_L, pos_x)
                else:
                    fieldk_E = b1 * self.L2E(self, delr_L, pos_x)
            elif 'bL1' in self.model_name:
                bL1 = biases[0]
                if 'cs2' in self.model_name:
                    cs2 = biases[1]
                    fieldk_E = ( bL1 + cs2*self.k2_E ) * self.L2E(self, delr_L, pos_x) + self.L2E(self, b2*d2r_L +  bG2*G2r_L, pos_x)
                else:
                    fieldk_E = bL1 * self.L2E(self, delr_L, pos_x) + self.L2E(self, b2*d2r_L +  bG2*G2r_L, pos_x)
            elif 'quad' in self.model_name:
                b1 = biases[0]
                b2 = biases[1]
                bG2 = biases[2]
                d2r_L = self.d2r(delr_L)
                G2r_L = self.G2r(delr_L)
                if 'cs2' in self.model_name:
                    cs2 = biases[3]
                    fieldk_E = ( b1 + cs2*self.k2_E ) * self.L2E(self, delr_L, pos_x) + self.L2E(self, 1., pos_x)
                else:
                    fieldk_E = bL1 * self.L2E(self, delr_L, pos_x) + self.L2E(self, 1., pos_x)
            elif 'cubic' in self.model_name:
                b1 = biases[0]
                b2 = biases[1]
                bG2 = biases[2]
                bGamma3 = biases[3]
                if self.rsd_flag == True:
                    pk_lin = vals[1]
                else:
                    pk_lin = vals[0]
                pddk_E, p1Gamma3k_E, p1S3k_E = self.transfer_function(pk_lin)
                p1Gamma3k_E /= pddk_E
                p1S3k_E /= pddk_E
                d2r_L = self.d2r(delr_L)
                G2r_L = self.G2r(delr_L)
                if 'cs2' in self.model_name:
                    cs2 = biases[4]
                    fieldk_E = ( b1 + cs2*self.k2_E + bGamma3*p1Gamma3k_E + b1*p1S3k_E ) * self.L2E(self, delr_L, pos_x) + self.L2E(self, b2*d2r_L +  bG2*G2r_L, pos_x)
                    if 'c1' in self.model_name:
                        c1 = biases[5]
                        fieldk_E = ( b1 + cs2*self.k2_E + c1*self.k2_E*self.mu2_E + bGamma3*p1Gamma3k_E + b1*p1S3k_E ) * self.L2E(self, delr_L, pos_x) + self.L2E(self, b2*d2r_L +  bG2*G2r_L, pos_x)
                else:
                    fieldk_E = ( b1 + bGamma3*p1Gamma3k_E + b1*p1S3k_E ) * self.L2E(self, delr_L, pos_x) + self.L2E(self, b2*d2r_L +  bG2*G2r_L, pos_x)

        return fieldk_E
