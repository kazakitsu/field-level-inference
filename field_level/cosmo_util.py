# !/usr/bin/env python3

from jax import jit
import field_level.util as util

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

@jit
def pow_Pk(k, norm=2e4, n_pow=-2.5):
    return norm*(k/0.1)**n_pow
