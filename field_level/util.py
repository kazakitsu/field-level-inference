# !/usr/bin/env python3
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from typing import Optional, Union

def loginterp_jax(x, y, yint=None, side="both", lp=1, rp=-2):
    if side == "both":
        side = "lr"

    #if jnp.sign(y[lp]) == jnp.sign(y[lp - 1]) and jnp.sign(y[lp]) == jnp.sign(y[lp + 1]):
    #    l = lp
    #else:
    #    l = lp + 2

    #if jnp.sign(y[rp]) == jnp.sign(y[rp - 1]) and jnp.sign(y[rp]) == jnp.sign(y[rp + 1]):
    #    r = rp
    #else:
    #    r = rp - 2
    l = lp
    r = rp

    dxl = x[l+1] - x[l]; lneff = jnp.gradient(y, dxl)[l] * x[l] / y[l]
    dxr = x[r] - x[r-1]; rneff = jnp.gradient(y, dxr)[r] * x[r] / y[r]

    def yint2(xx):
        left_cond = xx <= x[l]
        right_cond = xx >= x[r]
        middle_cond = jnp.logical_and(xx > x[l], xx < x[r])

        left_val = y[l] * jnp.nan_to_num((xx / x[l]) ** lneff)
        right_val = y[r] * jnp.nan_to_num((xx / x[r]) ** rneff)
        middle_val = jnp.interp(xx, x, y)

        return jax.lax.select(left_cond, left_val,
                              jax.lax.select(right_cond, right_val, middle_val))

    return yint2


@partial(jit, static_argnames=("axis",))
def trapezoid(
    y: jnp.ndarray,
    x: Optional[jnp.ndarray] = None,
    dx: Union[float, jnp.ndarray] = 1.0,
    axis: int = -1,
) -> jnp.ndarray:
    """
    Compute the trapezoidal rule integration along a given axis.

    Parameters
    ----------
    y : jnp.ndarray
        The values to integrate.
    x : jnp.ndarray or None, optional
        The sample points corresponding to `y`. If `x` is None (default),
        then the integration assumes a uniform spacing `dx`.
        If `x` is 1D, it is treated as the sample points for `y` along the given axis.
        If `x` has the same shape as `y`, then we take consecutive differences along `axis`.
    dx : float or jnp.ndarray, optional
        The uniform spacing between points if `x` is None.
    axis : int, optional
        The axis along which to integrate. Default is -1.

    Returns
    -------
    jnp.ndarray
        The result of trapezoidal integration of `y` along `axis`.
    """
    # Move the integration axis to the last dimension for easier slicing:
    y_arr = jnp.moveaxis(jnp.asarray(y, dtype=jnp.float32), axis, -1)

    if x is None:
        # uniform spacing
        dx_array = jnp.asarray(dx, dtype=jnp.float32)
        result = 0.5 * dx_array * (y_arr[..., 1:] + y_arr[..., :-1]).sum(axis=-1)
    else:
        # non-uniform spacing from x
        x_arr = jnp.moveaxis(jnp.asarray(x, dtype=jnp.float32), axis, -1)
        dx_array = jnp.diff(x_arr, axis=-1)
        result = 0.5 * (dx_array * (y_arr[..., 1:] + y_arr[..., :-1])).sum(axis=-1)

    return result


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
