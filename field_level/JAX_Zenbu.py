import jax.numpy as jnp
from jax import jit
from jax import lax
from functools import partial

import os

from scipy.interpolate import interp1d

from field_level.Zenbu_utils.spherical_bessel_transform_ncol_jax import SphericalBesselTransform as SphericalBesselTransform_ncol
from field_level.Zenbu_utils.spherical_bessel_transform import SphericalBesselTransform as SphericalBesselTransform
from field_level.Zenbu_utils.qfuncfft_update_pk_slim import QFuncFFT_JAX as QFuncFFT
from field_level.Zenbu_utils.loginterp_jax import loginterp_jax as loginterp

class Zenbu:
    '''
    Class to calculate power spectra up to one loop.
    
    Based on velocileptors
    
    https://github.com/sfschen/velocileptors/blob/master/LPT/cleft_fftw.py
    
    The bias parameters are ordered in pktable as
    1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2.
    Note that these are the component spectra (b_i, b_j) and not the coefficient multiplying b_i b_j in the auto.
    
    Can combine into a full one-loop real-space power spectrum using the function combine_bias_terms_pk.
    
    '''

    def __init__(self, kint, pint, kmin=1e-3, kmax=3, nk=100, kvec=None, cutoff=10, jn=5):

        
        self.cutoff = cutoff
        
        self.N  = len(kint)
        self.kint =  kint
        self.qint =  jnp.flip(1/kint)
        
        self.sph_obj1 = SphericalBesselTransform(self.kint, L=5, low_ring=True, fourier=True)
        self.sph_obj2 = SphericalBesselTransform(self.qint, L=5, low_ring=True, fourier=False)
        
        self.update_power_spectrum(pint)
        
        #self.pktable = None
        self.num_power_components = 4

        self.jn = jn
        self.sph = SphericalBesselTransform_ncol(self.qint, L=self.jn, ncol=self.num_power_components)
        
        if kvec is None:
            self.kmin, self.kmax = kmin, kmax
            self.nk = nk
    
            self.kv = jnp.logspace( jnp.log10(self.kmin),  jnp.log10(self.kmax), self.nk)
            #self.pktable =  jnp.zeros([self.nk, self.num_power_components+1])
            #self.pktable = self.pktable.at[:,0].set(self.kv)
        else:
            self.kmin, self.kmax = kvec[0], kvec[-1]
            self.nk = len(kvec)
            
            self.kv = kvec
            #self.pktable =  jnp.zeros([self.nk, self.num_power_components+1])
            #self.pktable = self.pktable.at[:,0].set(self.kv)

    def update_power_spectrum(self, p):
        # Updates the power spectrum and various q functions. Can continually compute for new cosmologies without reloading FFTW
        self.pint = p *  jnp.exp(-(self.kint/self.cutoff)**2)
        self.setup_powerspectrum()

    def setup_powerspectrum(self):
                
        self.qf = QFuncFFT(self.kint, self.pint, self.qint, self.sph_obj1, self.sph_obj2, oneloop=True, shear=True, third_order=True)
        
        # linear terms
        self.Xlin = self.qf.Xlin
        self.Ylin = self.qf.Ylin
        
        self.XYlin = self.Xlin + self.Ylin; self.sigma = self.XYlin[-1]
        self.yq = self.Ylin / self.qint
        
        self.Ulin = self.qf.Ulin
        self.corlin = self.qf.corlin
        
        self.Psi2graddelta = self.qf.xi_l_n(0,0,3./7*(self.qf.R1+self.qf.R2))
        self.theta = self.qf.theta

    def p_integrals(self, k):
        '''
        Compute P(k) for a single k as a vector of all bias contributions.
        
        '''
        ksq = k**2; kcu = k**3; k4 = k**4
        expon =  jnp.exp(-0.5*ksq * (self.XYlin - self.sigma))
        exponm1 =  jnp.expm1(-0.5*ksq * (self.XYlin - self.sigma))
        suppress =  jnp.exp(-0.5 * ksq  * self.sigma)
        
        ret =  jnp.zeros(self.num_power_components)
        
        bias_integrands =  jnp.zeros( (self.num_power_components,self.N)  )
        
        for l in range(self.jn):
            # l-dep functions
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            #mu3fac = (1. - 2.*(l-1)/ksq/self.Ylin / D**2) * mu1fac # mu3 terms start at j1 so l -> l-1
            #mu4fac = 1 - 4*l/ksq/self.Ylin/D**2 + 4*l*(l-1)/(ksq*self.Ylin*D**2)**2
            
            bias_integrands = bias_integrands.at[0,:].set(self.corlin -  ksq*mu2fac*self.Ulin**2) # (delta, delta)
            bias_integrands = bias_integrands.at[1,:].set(self.theta) # (delta, b3)
            bias_integrands = bias_integrands.at[2,:].set(self.Psi2graddelta) # (delta, k.Psi2 nabla delta)

            bias_integrands = bias_integrands.at[-1,:].set(1) # matter Zeldovich

            # Multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands = bias_integrands - bias_integrands[:, -1][:, None]  # Note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * (self.yq) ** l
                
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            
            ii = jnp.where( (ktemps - k) > 0, size=self.N)[0][0]
            kl, kr = ktemps[ii-1], ktemps[ii]
            bias_ffts_interp = (bias_ffts[:,ii-1] * (kr-k) + bias_ffts[:,ii] * (k - kl))/(kr-kl)
            ret += k**l * bias_ffts_interp
            
        return 4*suppress* jnp.pi*ret

    def make_ptable(self):

        def _ptable_loopfunc(ii, tab):
            return tab.at[ii,1:].set(self.p_integrals(self.kv[ii]))
   
        pktable =  jnp.zeros([self.nk, self.num_power_components+1])
        pktable = pktable.at[:,0].set(self.kv)

        pktable = lax.fori_loop(0,self.nk,_ptable_loopfunc, pktable)
    
        return pktable
