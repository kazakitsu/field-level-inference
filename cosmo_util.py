# !/usr/bin/env python3

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from Zenbu_utils.loginterp_jax import loginterp_jax

import jax
jax.config.update("jax_enable_x64", True)

import util

@jit
def growth_D(z, OM0):
    w = -1
    a = 1./(1+z)
    qa = ((1-OM0)/OM0)*a**(-3*w)
    return a*util.hyp2f1(-1./(3*w), (w-1)/(2*w), 1-5/(6*w), -qa)/(util.hyp2f1(-1./(3*w), (w-1)/(2*w), 1-5/(6*w), -(1-OM0)/OM0)) #normalized by D(z=0)

def sigma8_integ(ln_k):
    k = jnp.exp(ln_k)
    x = k*8.0
    wk = 3.*(jnp.sin(x)/x/x - jnp.cos(x)/x)/x
    return k**3*func_pk(k)*wk**2/(2.*jnp.pi**2)

@jit
def sigma8_simps(k, pk):  ### k [h/Mpc] and Pk [Mpc/h]^3
    ln_k = jnp.log(k)
    func_pk = loginterp_jax(k, pk)
    return jnp.sqrt(util.romb(sigma8_integ,
                              jnp.log(ln_k[0]), jnp.log(ln_k[-1]),
                              args=(func_pk) ))

@jit
def sigma8_romb(ln_k, pk):
    k = jnp.exp(ln_k)
    x = k*8.0
    win = 3.*(jnp.sin(x)/x/x - jnp.cos(x)/x)/x
    return jnp.sqrt(jax.scipy.integrate.trapezoid(k**3*pk*win**2/(2.*jnp.pi**2), ln_k))

@jit
def growth_f_fitting(z, OM0):
    a = 1./(1+z)
    a3 = a*a*a
    Ea = OM0/a3 + (1.-OM0)
    OM_a = OM0/a3/Ea
    return OM_a**(5./9.)

@jit
def pow_Pk(k, norm=2e4, n_pow=-2.5):
    return norm*(k/0.1)**n_pow
