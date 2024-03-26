# !/usr/bin/env python3
from __future__ import annotations

import jax
import numpy as np
#import scipy.integrate
import jax.numpy as jnp
from jax import jit
from functools import partial
import jax
from jax import vmap

from jax._src.numpy import util
from jax._src.typing import Array, ArrayLike

jax.config.update("jax_enable_x64", True)

#@util.implements(scipy.integrate.trapezoid)
@partial(jit, static_argnames=('axis',))
def trapezoid(y: ArrayLike, x: ArrayLike | None = None, dx: ArrayLike = 1.0,
              axis: int = -1) -> Array:
  # TODO(phawkins): remove this annotation after fixing jnp types.
  dx_array: Array
  if x is None:
    util.check_arraylike('trapezoid', y)
    y_arr, = util.promote_dtypes_inexact(y)
    dx_array = jnp.asarray(dx)
  else:
    util.check_arraylike('trapezoid', y, x)
    y_arr, x_arr = util.promote_dtypes_inexact(y, x)
    if x_arr.ndim == 1:
      dx_array = jnp.diff(x_arr)
    else:
      dx_array = jnp.moveaxis(jnp.diff(x_arr, axis=axis), axis, -1)
  y_arr = jnp.moveaxis(y_arr, axis, -1)
  return 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).sum(-1)




'''
from https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/integrate.html
'''

# Romberg quadratures for numeric integration.
#
# Written by Scott M. Ransom <ransom@cfa.harvard.edu>
# last revision: 14 Nov 98
#
# Cosmetic changes by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 1999-7-21
#
# Adapted to scipy by Travis Oliphant <oliphant.travis@ieee.org>
# last revision: Dec 2001

def _difftrap1(function, interval):
    """
    Perform part of the trapezoidal rule to integrate a function.
    Assume that we had called difftrap with all lower powers-of-2
    starting with 1.  Calling difftrap only returns the summation
    of the new ordinates.  It does _not_ multiply by the width
    of the trapezoids.  This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    return 0.5 * (function(interval[0]) + function(interval[1]))


def _difftrapn(function, interval, numtraps):
    """
    Perform part of the trapezoidal rule to integrate a function.
    Assume that we had called difftrap with all lower powers-of-2
    starting with 1.  Calling difftrap only returns the summation
    of the new ordinates.  It does _not_ multiply by the width
    of the trapezoids.  This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    numtosum = numtraps // 2
    h = (1.0 * interval[1] - 1.0 * interval[0]) / numtosum
    lox = interval[0] + 0.5 * h
    points = lox + h * np.arange(0, numtosum)
    s = np.sum(function(points))
    return s


def _romberg_diff(b, c, k):
    """
    Compute the differences for the Romberg quadrature corrections.
    See Forman Acton's "Real Computing Made Real," p 143.
    """
    tmp = 4.0**k
    return (tmp * c - b) / (tmp - 1.0)


def romb(function, a, b, args=(), divmax=6, return_error=False):
    """
    Romberg integration of a callable function or method.
    Returns the integral of `function` (a function of one variable)
    over the interval (`a`, `b`).
    If `show` is 1, the triangular array of the intermediate results
    will be printed.  If `vec_func` is True (default is False), then
    `function` is assumed to support vector arguments.
    Parameters
    ----------
    function : callable
        Function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    Returns
    -------
    results  : float
        Result of the integration.
    Other Parameters
    ----------------
    args : tuple, optional
        Extra arguments to pass to function. Each element of `args` will
        be passed as a single argument to `func`. Default is to pass no
        extra arguments.
    divmax : int, optional
        Maximum order of extrapolation. Default is 10.
    See Also
    --------
    fixed_quad : Fixed-order Gaussian quadrature.
    quad : Adaptive quadrature using QUADPACK.
    dblquad : Double integrals.
    tplquad : Triple integrals.
    romb : Integrators for sampled data.
    simps : Integrators for sampled data.
    cumtrapz : Cumulative integration for sampled data.
    ode : ODE integrator.
    odeint : ODE integrator.
    References
    ----------
    .. [1] 'Romberg's method' http://en.wikipedia.org/wiki/Romberg%27s_method
    Examples
    --------
    Integrate a gaussian from 0 to 1 and compare to the error function.
    >>> from scipy import integrate
    >>> from scipy.special import erf
    >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
    >>> result = integrate.romberg(gaussian, 0, 1, show=True)
    Romberg integration of <function vfunc at ...> from [0, 1]
    ::
       Steps  StepSize  Results
           1  1.000000  0.385872
           2  0.500000  0.412631  0.421551
           4  0.250000  0.419184  0.421368  0.421356
           8  0.125000  0.420810  0.421352  0.421350  0.421350
          16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
          32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350
    The final result is 0.421350396475 after 33 function evaluations.
    >>> print("%g %g" % (2*result, erf(1)))
    0.842701 0.842701
    """
    vfunc = jit(lambda x: function(x, *args))

    n = 1
    interval = [a, b]
    intrange = b - a
    ordsum = _difftrap1(vfunc, interval)
    result = intrange * ordsum
    state = np.repeat(np.atleast_1d(result), divmax + 1, axis=-1)
    err = np.inf

    def scan_fn(carry, y):
        x, k = carry
        x = _romberg_diff(y, x, k + 1)
        return (x, k + 1), x

    for i in range(1, divmax + 1):
        n = 2**i
        ordsum = ordsum + _difftrapn(vfunc, interval, n)

        x = intrange * ordsum / n
        _, new_state = jax.lax.scan(scan_fn, (x, 0), state[:-1])

        new_state = np.concatenate([np.atleast_1d(x), new_state])

        err = np.abs(state[i - 1] - new_state[i])
        state = new_state

    if return_error:
        return state[i], err
    else:
        return state[i]


def simps(f, a, b, N=128):
    """Approximate the integral of f(x) from a to b by Simpson's rule.

    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.

    Examples
    --------
    >>> simps(lambda x : 3*x**2,0,1,10)
    1.0

    Notes:
    ------
    Stolen from: https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/
    """
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S


'''
from https://github.com/google/jax/issues/2991
'''

@jit
def hyp2f1(a, b, c, z, n_max_iter=500, tol=1e-10, debug_mode=False):
    """
    Hypergeometric function implemented in jax, based on John Pearson's MSc thesis
    Computation of Hypergeometric Functions, specifically taylora2f1
    :param a: a (1D jax device array)
    :param b: b (1D jax device array)
    :param c: c (1D jax device array)
    :param z: z (1D jax device array), either one value of z for each (a, b, c) or different z-values for a single (a, b, c)
    :param n_max_iter: maximum number of iterations
    :param tol: tolerance
    :param debug_mode: debugging mode, doesn't use jitted operations
    :return: Gauss hypergeometric function 1F2(a, b, c, z)
    """

    a, b, c, z = jnp.atleast_1d(a), jnp.atleast_1d(b), jnp.atleast_1d(c), jnp.atleast_1d(z)
    assert a.ndim == b.ndim == c.ndim == z.ndim == 1

    # If only a single value for a, b, c is provided, but multiple values of z: tile a, b, c
    if a.shape[0] == b.shape[0] == c.shape[0] == 1 and z.shape[0] > 1:
        a = a * jnp.ones_like(z)
        b = b * jnp.ones_like(z)
        c = c * jnp.ones_like(z)

    assert a.shape[0] == b.shape[0] == c.shape[0] == z.shape[0], f"{a.shape[0], b.shape[0], c.shape[0], z.shape[0]}"

    def compute_output(a_loc, b_loc, c_loc, z_loc):

        def cond_fun(val):
            step_, a_, b_, c_, z_, coeff_old_, coeff_new_, sum_new_, tol_, n_max_iter_ = val
            cond_steps = jnp.all(step_ < n_max_iter_)
            cond_tol = jnp.all(jnp.logical_or(jnp.abs(coeff_old_) / jnp.abs(sum_new_) > tol_,
                                              jnp.abs(coeff_new_) / jnp.abs(sum_new_) > tol_))
            return jnp.all(jnp.logical_and(cond_steps, cond_tol))

        def body_fun(val):
            step_, a_, b_, c_, z_, coeff_old_, coeff_new_, sum_new_, tol_, n_max_iter_ = val
            coeff_new_ = (a_ + step_ - 1.0) * (b_ + step_ - 1.0) / (c_ + step_ - 1.0) * z_ / step_ * coeff_old_
            sum_new_ += coeff_new_
            return step_ + 1, a_, b_, c_, z_, coeff_new_, coeff_new_, sum_new_, tol_, n_max_iter_

        init_val = (jnp.asarray(1.0, ), a_loc * 1.0, b_loc * 1.0, c_loc * 1.0, z_loc * 1.0, jnp.asarray(1.0, ),
                    a_loc * 1.0, jnp.asarray(1.0, ), jnp.asarray(tol), jnp.asarray(n_max_iter * 1.0))

        if debug_mode:
            def while_loop(cond_fun, body_fun, init_val):
                val = init_val
                while cond_fun(val):
                    val = body_fun(val)
                return val

        else:
            while_loop = jax.lax.while_loop

        final_step, _, _, _, _, _, _, sum_out, _, _ = while_loop(cond_fun, body_fun, init_val=init_val)
        return sum_out

    def gamma(val):
        return jnp.where(val >= 0, jnp.exp(jax.scipy.special.gammaln(val)), -jnp.exp(jax.scipy.special.gammaln(val)))

    # Set indices for the different cases of the transformation mapping z to within the radius rho < 0.5
    # where Taylor series converges fast
    cases = jnp.where(z < -1,
                      1, jnp.where(z < 0,
                                   2, jnp.where(z <= 0.5,
                                                3, jnp.where(z <= 1,
                                                             4, jnp.where(z <= 2,
                                                                          5, 6)))))

    cases -= 1  # 0-based indexing

    # Define the branches
    def branch_1(this_a, this_b, this_c, this_z):
        term_1 = (1 - this_z) ** (-this_a) * (
                gamma(this_c) * gamma(this_b - this_a) / gamma(this_b) / gamma(this_c - this_a)) \
                 * compute_output(this_a, this_c - this_b, this_a - this_b + 1.0, 1.0 / (1.0 - this_z))
        term_2 = (1 - this_z) ** (-this_b) * (
                gamma(this_c) * gamma(this_a - this_b) / gamma(this_a) / gamma(this_c - this_b)) \
                 * compute_output(this_b, this_c - this_a, this_b - this_a + 1.0, 1.0 / (1.0 - this_z))
        return term_1 + term_2

    def branch_2(this_a, this_b, this_c, this_z):
        return (1 - this_z) ** (-this_a) * compute_output(this_a, this_c - this_b, this_c,
                                                          this_z / (this_z - 1.0))

    def branch_3(this_a, this_b, this_c, this_z):
        return compute_output(this_a, this_b, this_c, this_z)

    def branch_4(this_a, this_b, this_c, this_z):
        term_1 = (gamma(this_c) * gamma(this_c - this_a - this_b)
                  / gamma(this_c - this_a) / gamma(this_c - this_b)) \
                 * compute_output(this_a, this_b, this_a + this_b - this_c + 1.0, 1.0 - this_z)
        term_2 = (1 - this_z) ** (this_c - this_a - this_b) * \
                 (gamma(this_c) * gamma(this_a + this_b - this_c) / gamma(this_a) / gamma(this_b)) \
                 * compute_output(this_c - this_a, this_c - this_b, this_c - this_a - this_b + 1.0, 1.0 - this_z)
        return term_1 + term_2

    def branch_5(this_a, this_b, this_c, this_z):
        term_1 = this_z ** (-this_a) * (gamma(this_c) * gamma(this_c - this_a - this_b)
                                        / gamma(this_c - this_a) / gamma(this_c - this_b)) \
                 * compute_output(this_a, this_a - this_c + 1.0, this_a + this_b - this_c + 1.0, 1.0 - 1.0 / this_z)
        term_2 = this_z ** (this_a - this_c) * (1 - this_z) ** (this_c - this_a - this_b) \
                 * (gamma(this_c) * gamma(this_a + this_b - this_c) / gamma(this_a) / gamma(this_b)) \
                 * compute_output(this_c - this_a, 1.0 - this_a, this_c - this_a - this_b + 1.0, 1.0 - 1.0 / this_z)
        return term_1 + term_2

    def branch_6(this_a, this_b, this_c, this_z):
        term_1 = (-this_z) ** (-this_a) * (
                gamma(this_c) * gamma(this_b - this_a) / gamma(this_b) / gamma(this_c - this_a)) \
                 * compute_output(this_a, this_a - this_c + 1.0, this_a - this_b + 1.0, 1.0 / this_z)
        term_2 = (-this_z) ** (-this_b) * (
                gamma(this_c) + gamma(this_a - this_b) / gamma(this_a) / gamma(this_c - this_b)) \
                 * compute_output(this_b - this_c + 1.0, this_b, this_b - this_a + 1.0, 1.0 / this_z)
        return term_1 + term_2

    branches = [branch_1, branch_2, branch_3, branch_4, branch_5, branch_6]

    def single_computation(val, branches_):
        case_, a_, b_, c_, z_ = val

        if debug_mode:
            def switch_fun(index, branches, *operands):
                index = jnp.clip(0, index, len(branches) - 1)
                return branches[index](*operands)

        else:
            switch_fun = jax.lax.switch

        return switch_fun(case_, branches_, a_, b_, c_, z_)

    # Compute outputs
    if debug_mode:
        def map_fun(f, xs):
            return np.stack([f(x) for x in zip(*xs)])
    else:
        map_fun = jax.lax.map

    all_outputs = map_fun(lambda val: single_computation(val, branches), (cases, a, b, c, z))

    return all_outputs
