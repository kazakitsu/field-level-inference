import jax.numpy as jnp
#from jax.scipy.special import gammaln
from scipy.special import loggamma, gamma
import numpy as np
from jax import jit

def clog(z):
    """
    Compute the complex logarithm of a complex number in JAX.

    Args:
    z (complex): A complex number.

    Returns:
    complex: The complex logarithm of z.
    """
    return jnp.log(jnp.abs(z)) + 1j*jnp.angle(z)


g = 7
lanczos_coeffs  = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]

def cgamma(z):
    z -= 1
    x = lanczos_coeffs[0]
    for i in range(1, g+2):
        x += lanczos_coeffs[i]/(z+i)
    t = z + g + 0.5
    return jnp.sqrt(2*jnp.pi) * t**(z+0.5) * jnp.exp(-t) * x

def cloggamma(z):
    return clog(cgamma(z))

class SphericalBesselTransform:
    
    def __init__(self, qs, L=15, low_ring=True, fourier=False):
        '''
        Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
        the untransformed coordinate, up to a given order L in bessel functions (j_l for l
        less than or equal to L. The point is to save time by evaluating the Mellin transforms
        u_m in advance.
        
        Does not use fftw as in spherical_bessel_transform_fftw.py, which makes it convenient
        to evaluate the generalized correlation functions in qfuncfft, as there aren't as many
        ffts as in LPT modules so time saved by fftw is minimal when accounting for the
        startup time of pyFFTW.
        
        Based on Yin Li's package mcfit (https://github.com/eelregit/mcfit)
        with the above modifications.
        '''
        
        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = jnp.sqrt(jnp.pi)
        else:
            self.sqrtpi = jnp.sqrt(jnp.pi) / (2*jnp.pi**2)
        
        self.q = qs
        self.L = L
        
        self.Nx = len(qs)
        self.Delta = jnp.log(qs[-1]/qs[0])/(self.Nx-1)

        self.N = 2**(jnp.ceil(jnp.log2(self.Nx)).astype(int) + 1)
        self.Npad = self.N - self.Nx
        self.pads = jnp.zeros( (self.N-self.Nx)//2  )
        self.pad_iis = jnp.arange(self.Npad - self.Npad//2, self.N - self.Npad//2)
        
        # Set up the FFTLog kernels u_m up to, but not including, L
        ms = jnp.arange(0, self.N//2+1)
        self.ydict = {}; self.udict = {}; self.qdict= {}
        
        if low_ring:
            for ll in range(L):
                q = max(0, 1.5 - ll)
                lnxy = self.Delta/jnp.pi * jnp.angle(self.UK(ll,q+1j*jnp.pi/self.Delta)) #ln(xmin*ymax)
                ys = jnp.exp( lnxy - self.Delta) * qs/ (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * jnp.pi / self.N / self.Delta * ms) \
                        * jnp.exp(-2j * jnp.pi * lnxy / self.N / self.Delta * ms)

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q
        
        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(L):
                q = max(0, 1.5 - ll)
                ys = jnp.exp(-self.Delta) * qs / (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * jnp.pi / self.N / self.Delta * ms)

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q


    def sph(self, nu, fq):
        '''
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates
        self.q.
        '''
        q = self.qdict[nu]; y = self.ydict[nu]
        f = jnp.concatenate( (self.pads, self.q**(3-q)*fq, self.pads) )

        fks = jnp.fft.rfft(f)
        gks = self.udict[nu] * fks
        gs = jnp.fft.hfft(gks) / self.N

        return y, y**(-q) * gs[self.pad_iis]
    
    
    def UK(self, nu, z):
        '''
        The Mellin transform of the spherical bessel transform.
        '''

        return self.sqrtpi * jnp.exp(np.log(2)*(z-2) + loggamma(0.5*(nu+z)) - loggamma(0.5*(3+nu-z)))
        #return self.sqrtpi * jnp.exp(np.log(2)*(z-2)) * gamma(0.5*(nu+z)) / gamma(0.5*(3+nu-z))

    def update_tilt(self,nu,tilt):
        '''
        Update the tilt for a particular nu. Assume low ring coordinates.
        '''
        q = tilt; ll = nu
    
        ms = jnp.arange(0, self.N//2+1)
        lnxy = self.Delta/jnp.pi * jnp.angle(self.UK(ll,q+1j*self.pi/self.Delta)) #ln(xmin*ymax)
        ys = jnp.exp( lnxy - self.Delta) * self.q/ (self.q[0]*self.q[-1])
        us = self.UK(ll, q + 2j * jnp.pi / self.N / self.Delta * ms) \
            * jnp.exp(-2j * jnp.pi * lnxy / self.N / self.Delta * ms)
            
        self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q
