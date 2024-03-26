import jax
import jax.numpy as jnp

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
