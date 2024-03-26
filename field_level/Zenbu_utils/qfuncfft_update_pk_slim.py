import jax.numpy as jnp

from field_level.Zenbu_utils.loginterp_jax import loginterp_jax
from field_level.Zenbu_utils.spherical_bessel_transform import SphericalBesselTransform


class QFuncFFT_JAX:
    '''
       Class to calculate all the functions of q, X(q), Y(q), U(q), xi(q) etc.
       as well as the one-loop terms Q_n(k), R_n(k) using FFTLog.
       
       Throughout we use the ``generalized correlation function'' notation of 1603.04405.
       
       This is modified to take an IR scale kIR
              
       Note that one should always cut off the input power spectrum above some scale.
       I use exp(- (k/20)^2 ) but a cutoff at scales twice smaller works equivalently,
       and probably beyond that. The important thing is to keep all integrals finite.
       This is done automatically in the Zeldovich class.
       
       Currently using the numpy version of fft. The FFTW takes longer to start up and
       the resulting speedup is unnecessary in this case.
       
    '''
    def __init__(self, k, p, qv, sph, sphr, kIR = None, oneloop = True, shear = True, third_order = True, low_ring=True):

        self.oneloop = oneloop
        self.shear = shear
        self.third_order = third_order
        
        self.k = k
        self.p = p
        self.qv = qv
        
        if kIR is not None:
            self.ir_less = jnp.exp(- (self.k/kIR)**2 )
            self.ir_greater = -jnp.expm1(- (self.k/kIR)**2)
        else:
            self.ir_less = 1
            self.ir_greater = 0
        
        self.sph = sph
        self.sphr = sphr
        
        self.setup_xiln()
        self.setup_2pts()
        
        if self.shear:
            self.setup_shear()
        
        if self.oneloop:
            self.setup_QR()
            self.setup_oneloop_2pts()
            
        if self.third_order:
            self.setup_third_order()

    def setup_xiln(self):
        
        # Compute a bunch of generalized correlation functions
        self.xi00 = self.xi_l_n(0,0)
        self.xi1m1 = self.xi_l_n(1,-1)
        self.xi0m2 = self.xi_l_n(0,-2, side='right') # since this approaches constant on the left only interpolate on right
        self.xi2m2 = self.xi_l_n(2,-2)
        
        # Also compute the IR-cut lm2's
        self.xi0m2_lt = self.xi_l_n(0,-2, IR_cut = 'lt', side='right')
        self.xi2m2_lt = self.xi_l_n(2,-2, IR_cut = 'lt')
        
        #self.xi0m2_gt = self.xi_l_n(0,-2, IR_cut = 'gt', side='right')
        #self.xi2m2_gt = self.xi_l_n(2,-2, IR_cut = 'gt')
    
        # also compute those for one loop terms since they don't take much more time
        # also useful in shear terms
        self.xi20 = self.xi_l_n(2,0)
        self.xi40 = self.xi_l_n(4,0)
        
        self.xi11 = self.xi_l_n(1,1)
        self.xi31 = self.xi_l_n(3,1)
        self.xi3m1 = self.xi_l_n(3,-1)
        
        self.xi02 = self.xi_l_n(0,2)
        self.xi22 = self.xi_l_n(2,2)
    
    def setup_QR(self):
    
        _integrand_R1_0 = self.xi00/self.qv
        _integrand_R1_2 = self.xi20/self.qv
        _integrand_R1_4 = self.xi40/self.qv
        _integrand_R2_1 = self.xi1m1/self.qv
        _integrand_R2_3 = self.xi3m1/self.qv

        R1_0 = self.template_QR(0,_integrand_R1_0)
        R1_2 = self.template_QR(2,_integrand_R1_2)
        R1_4 = self.template_QR(4,_integrand_R1_4)
        R2_1 = self.template_QR(1,_integrand_R2_1)
        R2_3 = self.template_QR(3,_integrand_R2_3)

        self.R1 = self.k**2 * self.p * (8./15 * R1_0 - 16./21* R1_2 + 8./35 * R1_4)
        self.R2 = self.k**2 *self.p * (-2./15 * R1_0 - 2./21* R1_2 + 8./35 * R1_4 +  self.k * 2./5*R2_1 - self.k* 2./5*R2_3)

    def setup_2pts(self):
        # Piece together xi_l_n into what we need
        self.Xlin = 2./3 * (self.xi0m2[0] - self.xi0m2 - self.xi2m2)
        self.Ylin = 2 * self.xi2m2
        
        self.Xlin_lt = 2./3 * (self.xi0m2_lt[0] - self.xi0m2_lt - self.xi2m2_lt)
        self.Ylin_lt = 2 * self.xi2m2_lt
        
        self.Xlin_gt = self.Xlin - self.Xlin_lt
        self.Ylin_gt = self.Ylin - self.Ylin_lt
        
        #self.Xlin_gt = 2./3 * (self.xi0m2_gt[0] - self.xi0m2_gt - self.xi2m2_gt)
        #self.Ylin_gt = 2 * self.xi2m2_gt
        
        self.Ulin = - self.xi1m1
        self.corlin = self.xi00
    
    def setup_shear(self):
        return 0

    def setup_oneloop_2pts(self):
        return 0
    
    def setup_third_order(self):
        # All the terms involving the third order bias, which is really just a few
        
        P3_0 = self.k**2 * self.template_QR(0, 24./5*self.xi00/self.qv)
        P3_1 = self.k    * self.template_QR(1, -16./5*self.xi11/self.qv)
        P3_2 = self.k**2 * self.template_QR(2, -20./7*self.xi20/self.qv) + self.template_QR(2,4.*self.xi22/self.qv)
        P3_3 = self.k    * self.template_QR(3, -24./5*self.xi31/self.qv)
        P3_4 = self.k**2 * self.template_QR(4, 72./35*self.xi40/self.qv)

        self.Rb3 = 2 * 2./63 * (P3_0 + P3_1 + P3_2 + P3_3 + P3_4) * self.p
        
        self.theta = self.xi_l_n(0,0, _int= self.Rb3)
        #self.Ub3 = - self.xi_l_n(1,-1, _int= self.Rb3)
        
    
    def xi_l_n(self, l, n, _int=None, IR_cut = 'all', extrap=False, qmin=1e-3, qmax=1000, side='both'):
        '''
        Calculates the generalized correlation function xi_l_n, which is xi when l = n = 0
        
        If _int is None assume integrating the power spectrum.
        '''
        if _int is None:
            integrand = self.p * self.k**n
        else:
            integrand = _int * self.k**n
        
        if IR_cut != 'all':
            if IR_cut == 'gt':
                integrand *= self.ir_greater
            elif IR_cut == 'lt':
                integrand *= self.ir_less
        
        qs, xint =  self.sph.sph(l,integrand)

        if extrap:
            qrange = (qs > qmin) * (qs < qmax)
            return loginterp_jax(qs[qrange],xint[qrange],side=side)(self.qv)
        else:
            return jnp.interp(self.qv, qs, xint)

    def template_QR(self,l,integrand):
        '''
        Interpolates the Hankel transformed R(k), Q(k) back onto self.k
        '''
        kQR, QR = self.sphr.sph(l,integrand)
        return jnp.interp(self.k, kQR, QR)

