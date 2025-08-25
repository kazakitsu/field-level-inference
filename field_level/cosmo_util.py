# !/usr/bin/env python3
from functools import partial
from jax import jit
import jax.numpy as jnp
import field_level.util as util
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

@jit
def growth_D(z, OM0):
    w = -1
    a = 1./(1+z)
    qa = ((1-OM0)/OM0)*a**(-3*w)
    return a*util.hyp2f1(-1./(3*w), (w-1)/(2*w), 1-5/(6*w), -qa)/(util.hyp2f1(-1./(3*w), (w-1)/(2*w), 1-5/(6*w), -(1-OM0)/OM0)) #normalized by D(z=0)

@jit
def growth_f_fitting(z, OM0):
    a = 1./(1+z)
    a3 = a*a*a
    Ea = OM0/a3 + (1.-OM0)
    OM_a = OM0/a3/Ea
    return OM_a**(5./9.)

class Pk_Provider:
    """Small helper to supply [k, P(k)] in physical h/Mpc units for Base_Forward.linear_modes."""
    def __init__(self):
        self.cp = CPJ(probe='mpk_lin')

    @partial(jit, static_argnames=('self',))
    def pk_lin_table(self, cosmo_params):
        ### cosmo_params = omega_b, omega_c, hubble, ns, ln1010As, z
        x = jnp.asarray(cosmo_params)
        hubble = x[2]
        # Emulated modes are in 1/Mpc. Convert to h/Mpc.
        k_emu = self.cp.modes
        P_emu = self.cp.predict(x)
        k = k_emu / hubble
        P = P_emu * hubble**3
        return jnp.stack([k, P], axis=0).T  # shape (N, 2)

    @partial(jit, static_argnames=('self',))
    def pow_Pk(k, norm=2e4, n_pow=-2.5):
        return norm*(k/0.1)**n_pow

    @partial(jit, static_argnames=('self', 'R', 'integral_type'))
    def sigmaR(self, pk_lin, R=8.0, integral_type='trap'):
        if integral_type == 'trap':
            x = pk_lin[:,0] * R
            wk = 3.*(jnp.sin(x)/x/x - jnp.cos(x)/x)/x
            sigma8 = jnp.sqrt(util.trapezoid(pk_lin[:,0]**3*pk_lin[:,1]*wk**2/(2.*jnp.pi**2),
                                             jnp.log(pk_lin[:,0])) )
        return sigma8
